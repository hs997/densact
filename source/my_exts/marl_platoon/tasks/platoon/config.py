import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationTermCfg, TerminationTermCfg, \
    ObservationGroupCfg, EventTermCfg
import isaaclab.envs.mdp as mdp
# 引入长方体、材质、刚体属性配置
from isaaclab.sim import DomeLightCfg, UrdfFileCfg, SimulationCfg, PhysxCfg, \
    CuboidCfg, PreviewSurfaceCfg, RigidBodyMaterialCfg, RigidBodyPropertiesCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
# 注意：新版 IsaacLab 中 quat_rotate 可能被弃用，建议用 quat_apply 或 quat_rotate_inverse
from isaaclab.utils.math import quat_apply_inverse, quat_apply

# =============================================================================
# 0. 资产定义
# =============================================================================
USER_URDF_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/Melodic_wheeltec_robot_src_250707/src/turn_on_wheeltec_robot/urdf/mini_4wd_robot.urdf"

MY_CAR_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=USER_URDF_PATH,
        fix_base=False,
        make_instanceable=True,
        joint_drive=UrdfFileCfg.JointDriveCfg(
            drive_type="force", target_type="velocity",
            # 保持低阻尼，Sim-to-Sim 关键设置
            gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.05)
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    actuators={
        "all_wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0, damping=2.0, effort_limit_sim=100.0, velocity_limit_sim=10.0,
        ),
    }
)


# =============================================================================
# 1. 论文核心：Ground Truth 评估函数 (Outer Loop Objectives)
# =============================================================================

def reward_true_formation_success(env):
    """
    [论文 \bar{R}] 绝对成功指标 (Sparse Reward)
    只有当所有跟随车与前车的距离误差都 < 10cm 时，才算成功。
    用于评估智能体是否真正学会了完美编队。
    """
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    target_dist = 0.6
    tolerance = 0.2  # 苛刻的成功标准：10cm 误差

    all_success_mask = torch.ones(env.num_envs, device=env.device)

    for i in range(1, 4):
        dist = torch.norm(env.scene[robots[i - 1]].data.root_pos_w[:, :2] -
                          env.scene[robots[i]].data.root_pos_w[:, :2], dim=1)
        # 只要有一对车距离不达标，这一帧就算失败 (0分)
        is_good = (torch.abs(dist - target_dist) < tolerance).float()
        all_success_mask *= is_good

    return all_success_mask


def reward_true_collision_fail(env):
    """
    [论文 \bar{R}] 绝对失败指标 (Sparse Reward)
    只要发生任何碰撞 (距离 < 30cm)，就算严重事故。
    """
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    min_safe_dist = 0.30
    any_crash_mask = torch.zeros(env.num_envs, device=env.device)

    for i in range(1, 4):
        dist = torch.norm(env.scene[robots[i - 1]].data.root_pos_w[:, :2] -
                          env.scene[robots[i]].data.root_pos_w[:, :2], dim=1)
        is_crash = (dist < min_safe_dist).float()
        # 逻辑或：只要有一对撞了，就是撞了
        any_crash_mask = torch.max(any_crash_mask, is_crash)

    return any_crash_mask


# =============================================================================
# 2. 训练用逻辑函数 (Inner Loop Objectives)
# =============================================================================

def reward_leader_velocity_matching(env):
    """领航车速度跟随"""
    command_vel = env.command_manager.get_command("base_velocity")
    actual_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error = torch.norm(command_vel[:, :2] - actual_vel, dim=1)
    reward = torch.exp(-torch.square(error) / 0.2)

    if not hasattr(env, "_debug_leader_vel_once"):
        env._debug_leader_vel_once = True
        print("[DEBUG][leader_motion] command_vel[0]=", command_vel[0].detach().cpu().tolist())
        print("[DEBUG][leader_motion] actual_vel[0]=", actual_vel[0].detach().cpu().tolist())
        print("[DEBUG][leader_motion] error_mean=", float(error.mean().item()))
        print("[DEBUG][leader_motion] reward_mean=", float(reward.mean().item()))

    return reward


# =============================================================================
# 论文复现：动态自适应奖励 (Curriculum Reward)
# =============================================================================

def reward_platoon_dist_chain_dynamic(env):
    """
    [修复版] 动态队形奖励：强制直线编队
    修复逻辑：不仅计算距离，还分别计算 X轴(纵向) 和 Y轴(横向) 的误差。
    """
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    total_score = 0
    target_dist_x = 0.6  # 期望纵向距离 0.6m

    # 1. 动态 Sigma (保持之前的逻辑，用于从宽容变严厉)
    current_step = env.common_step_counter
    # 注意：这里的时间表可以根据你的实际步数调整，之前是50M，现在如果是Play模式其实只看最终效果
    # 训练时建议：start=2.0, end=0.2
    # Play时其实这个函数不起作用（Play只跑模型，不训练），但为了逻辑一致保持原样
    progress = min(current_step / 50_000_000.0, 1.0)
    current_sigma = 2.0 + (0.2 - 2.0) * progress

    for i in range(1, 4):
        # 获取两车在世界坐标系下的位置
        pos_leader = env.scene[robots[i - 1]].data.root_pos_w[:, :2]
        pos_follower = env.scene[robots[i]].data.root_pos_w[:, :2]
        quat_leader = env.scene[robots[i - 1]].data.root_quat_w

        # --- 核心修改：计算相对位置 (在领航车坐标系下) ---
        # 我们把后车的位置，转换到前车的坐标系里去
        # 结果是一个向量 (rel_x, rel_y)
        # rel_x: 后车在前车的前面(+)还是后面(-)
        # rel_y: 后车在前车的左边(+)还是右边(-)
        diff_vec = pos_follower - pos_leader
        # 注意：这里需要扩展 quat 的维度以匹配 diff_vec，或者 isaaclab 的工具函数能自动处理
        # 为保险起见，我们直接用数学公式或工具函数
        rel_pos = quat_apply_inverse(quat_leader, torch.cat([diff_vec, torch.zeros_like(diff_vec[:, 0:1])], dim=1))

        rel_x = rel_pos[:, 0]
        rel_y = rel_pos[:, 1]

        # --- 计算误差 ---
        # 1. 纵向误差：我们希望后车在前车后面 0.6m
        # 所以 rel_x 应该是 -0.6
        error_longitudinal = rel_x - (-target_dist_x)

        # 2. 横向误差：我们希望后车在正后方，不能偏左偏右
        # 所以 rel_y 应该是 0.0
        error_lateral = rel_y - 0.0

        # 3. 综合评分
        # 我们对横向误差惩罚得更重一点 (sigma/2)，强迫它们走直线
        score_long = torch.exp(-torch.square(error_longitudinal) / current_sigma)
        score_lat = torch.exp(-torch.square(error_lateral) / (current_sigma / 2.0))

        # 两个维度都好，才给高分
        total_score += score_long * score_lat

    return total_score / 3.0


def reward_lateral_penalty_hard(env):
    """
    [强力补丁] 专门惩罚横向偏差
    只要不在一条直线上，直接扣分！
    """
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    total_penalty = 0

    for i in range(1, 4):
        pos_leader = env.scene[robots[i - 1]].data.root_pos_w[:, :2]
        pos_follower = env.scene[robots[i]].data.root_pos_w[:, :2]
        quat_leader = env.scene[robots[i - 1]].data.root_quat_w

        # 计算相对位置
        diff_vec = pos_follower - pos_leader
        rel_pos = quat_apply_inverse(quat_leader, torch.cat([diff_vec, torch.zeros_like(diff_vec[:, 0:1])], dim=1))

        rel_y = rel_pos[:, 1]  # 横向偏差

        # 惩罚绝对值：偏离 10cm 就扣 0.1 * weight
        total_penalty += torch.abs(rel_y)

    return total_penalty  # 返回的是正数，我们在 config 里给它负权重

def reward_collision_risk(env):
    """[Dense Reward] 碰撞风险梯度惩罚"""
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    penalty = 0
    min_safe_dist = 0.35
    for i in range(1, 4):
        dist = torch.norm(env.scene[robots[i - 1]].data.root_pos_w[:, :2] -
                          env.scene[robots[i]].data.root_pos_w[:, :2], dim=1)
        is_too_close = (dist < min_safe_dist).float()
        penalty += is_too_close
    return penalty


def reward_move_backward_penalty(env):
    """禁止倒车"""
    penalty = 0
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    for name in robots:
        vel_x = env.scene[name].data.root_lin_vel_b[:, 0]
        is_backing = (vel_x < -0.05).float()
        penalty += is_backing
    return penalty


def reward_forward_drive_chain(env):
    """鼓励向前"""
    total = 0
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    for name in robots:
        vel_x = env.scene[name].data.root_lin_vel_b[:, 0]
        total += vel_x
    return total


def termination_bad_heading_chain(env):
    """检查掉头 (>90度)"""
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    any_bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    target_dir = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    for name in robots:
        quat = env.scene[name].data.root_quat_w
        # 使用 quat_apply 计算局部朝向
        heading_vec = quat_apply(quat, target_dir)
        is_backing = heading_vec[:, 0] < 0.0
        any_bad = any_bad | is_backing
    return any_bad


def obs_platoon_chain(env):
    """观测函数"""
    robots = ["robot", "robot_2", "robot_3", "robot_4"]
    obs_list = []
    for i in range(1, 4):
        pos_l = env.scene[robots[i - 1]].data.root_pos_w
        pos_f = env.scene[robots[i]].data.root_pos_w
        quat_f = env.scene[robots[i]].data.root_quat_w
        rel_pos_local = quat_apply_inverse(quat_f, pos_l - pos_f)
        vel_l = env.scene[robots[i - 1]].data.root_lin_vel_b[:, :2]
        vel_f = env.scene[robots[i]].data.root_lin_vel_b[:, :2]
        rel_vel = vel_l - vel_f
        obs_list.append(torch.cat([rel_pos_local[:, :2], rel_vel], dim=-1))
    return torch.cat(obs_list, dim=-1)


# =============================================================================
# 3. 配置组定义
# =============================================================================

@configclass
class PlatoonObservationsCfg:
    @configclass
    class PolicyGroupCfg(ObservationGroupCfg):
        rel_pos_chain = ObservationTermCfg(func=obs_platoon_chain)
        last_vel = ObservationTermCfg(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_4")})
        last_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_4")})

    policy: PolicyGroupCfg = PolicyGroupCfg()


@configclass
class PlatoonRewardsCfg:
    # --- Inner Loop: 训练引导信号 (Dense) ---
    leader_motion = RewardTermCfg(func=reward_leader_velocity_matching, weight=2.0)
    formation = RewardTermCfg(func=reward_platoon_dist_chain_dynamic, weight=4.0)
    collision_risk = RewardTermCfg(func=reward_collision_risk, weight=-10.0)
    no_backward = RewardTermCfg(func=reward_move_backward_penalty, weight=-5.0)
    forward_drive = RewardTermCfg(func=reward_forward_drive_chain, weight=0.5)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    lateral_correct = RewardTermCfg(func=reward_lateral_penalty_hard, weight=-2.0)

    # --- Outer Loop: 真实目标评估 (Sparse) ---
    # 我们给一个非零的小权重，这样它们会显示在 Tensorboard 日志里
    # 你可以通过这两个指标来判断你的 Dense Reward 设计得好不好
    true_success = RewardTermCfg(func=reward_true_formation_success, weight=0.1)
    true_fail = RewardTermCfg(func=reward_true_collision_fail, weight=-0.1)

    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)


@configclass
class PlatoonEnvCfg(ManagerBasedRLEnvCfg):
    @configclass
    class PlatoonAlgorithmCfg:
        """Task-local algorithm mode flags.

        PPO remains the default stable baseline. HAPPO/teacher/attack/shield are
        toggles for the task internals, not separate launch entrypoints.
        """

        algorithm: str = "ppo"
        enable_teacher: bool = False
        enable_attack: bool = False
        enable_shield: bool = False
        use_happo_layout: bool = False
        use_happo_actions: bool = False

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        # --- Robot 1 (领航车): 旗舰深蓝 ---
        robot = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        robot.init_state.pos = (1.0, 0.0, 0.5)
        robot.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.15, 0.4),  # 深蓝
            metallic=0.6, roughness=0.3  # 加一点金属反光感
        )

        # --- Robot 2 (跟随车): 珍珠皓白 ---
        robot_2 = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        robot_2.init_state.pos = (0.0, 0.0, 0.5)
        robot_2.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.9, 0.9),  # 亮白
            metallic=0.2, roughness=0.2  # 类似烤漆
        )

        # --- Robot 3 (跟随车): 太空银灰 ---
        robot_3 = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3")
        robot_3.init_state.pos = (-1.0, 0.0, 0.5)
        robot_3.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.55),  # 银灰带点蓝
            metallic=0.8, roughness=0.3  # 强金属质感
        )

        # --- Robot 4 (跟随车): 波尔多红 ---
        robot_4 = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_4")
        robot_4.init_state.pos = (-2.0, 0.0, 0.5)
        robot_4.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.4, 0.05, 0.05),  # 暗红
            metallic=0.5, roughness=0.4
        )

        # 2. 地形配置 (无限平面基底)
        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

        # 3. 高速公路实体资产
        highway_road = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Highway",
            spawn=CuboidCfg(
                size=(2000.0, 1.0, 0.1),  # 长2km，宽20m
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),  # 沥青色
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.2,
                    dynamic_friction=1.0,
                    restitution=0.0
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True
                )
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(990.0, 0.0, -0.05))
        )

        light = AssetBaseCfg(prim_path="/World/light", spawn=DomeLightCfg(intensity=3000.0))

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=25.0)
    observations: PlatoonObservationsCfg = PlatoonObservationsCfg()
    rewards: PlatoonRewardsCfg = PlatoonRewardsCfg()
    algorithm: PlatoonAlgorithmCfg = PlatoonAlgorithmCfg()

    @configclass
    class ActionCfg:
        joint_vel_1 = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=12.0)
        joint_vel_2 = mdp.JointVelocityActionCfg(asset_name="robot_2", joint_names=[".*_wheel_joint"], scale=12.0)
        joint_vel_3 = mdp.JointVelocityActionCfg(asset_name="robot_3", joint_names=[".*_wheel_joint"], scale=12.0)
        joint_vel_4 = mdp.JointVelocityActionCfg(asset_name="robot_4", joint_names=[".*_wheel_joint"], scale=12.0)

    actions: ActionCfg = ActionCfg()

    @configclass
    class CommandCfg:
        base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.5, 0.8),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0)
            )
        )

    commands: CommandCfg = CommandCfg()

    @configclass
    class TerminationsCfg:
        time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
        reset_on_bad_ori = TerminationTermCfg(func=termination_bad_heading_chain, time_out=False)

    terminations: TerminationsCfg = TerminationsCfg()

    # -------------------------------------------------------------------------
    # 论文对齐：攻击器/Teacher/安全投影 超参数入口（先作为配置入口）
    # 注：真正的 CA-GAN 与 meta-gradient 更新逻辑将在训练脚本中调用这些参数
    # -------------------------------------------------------------------------
    @configclass
    class AttackCfg:
        enable_attack: bool = True
        max_fdi_pos: float = 8.0      # epsilon_p
        max_fdi_acc: float = 3.0      # epsilon_a
        max_dos_rate: float = 0.30    # rho_max

    @configclass
    class TeacherCfg:
        enable_teacher: bool = True
        teacher_lr: float = 1.0e-4
        realism_lambda: float = 0.1

    @configclass
    class SafetyShieldCfg:
        enable_shield: bool = True
        d_crit: float = 0.50
        d_drop: float = 1.20
        brake_action: float = -1.0
        catchup_action: float = 0.35

    attack: AttackCfg = AttackCfg()
    teacher: TeacherCfg = TeacherCfg()
    safety_shield: SafetyShieldCfg = SafetyShieldCfg()

    sim: SimulationCfg = SimulationCfg(dt=0.005, use_fabric=True, render_interval=4)
    decimation: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0