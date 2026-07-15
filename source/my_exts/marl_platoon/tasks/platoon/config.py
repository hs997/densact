import torch
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationTermCfg, TerminationTermCfg, \
    ObservationGroupCfg, EventTermCfg
import isaaclab.envs.mdp as mdp
# 引入长方体、材质、刚体属性配置
from isaaclab.sim import DomeLightCfg, UrdfFileCfg, UsdFileCfg, SimulationCfg, PhysxCfg, \
    CuboidCfg, CylinderCfg, PreviewSurfaceCfg, MdlFileCfg, RigidBodyMaterialCfg, RigidBodyPropertiesCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
# 注意：新版 IsaacLab 中 quat_rotate 可能被弃用，建议用 quat_apply 或 quat_rotate_inverse
from isaaclab.utils.math import quat_apply_inverse, quat_apply

# =============================================================================
# 0. 资产定义
# =============================================================================
USER_URDF_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/Melodic_wheeltec_robot_src_250707/src/turn_on_wheeltec_robot/urdf/senior_4wd_bs_robot.urdf"
HIGHWAY_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/highway_straight/highway_straight.usda"
CITY_SHOP_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_straight/building_shop.usda"
CITY_OFFICE_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_straight/building_office_low.usda"
CITY_APARTMENT_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_straight/building_apartment.usda"
CITY_REAL_BUILDING_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_realistic/usd/Launch_Building_Coon_Building.usd"
CITY_REAL_FACADE_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_realistic/usd/Launch_Building_Coon_Exterior_Joint1_001.usd"
CITY_REAL_STATION_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_realistic/usd/Tower_Station_01_Art.usd"
CITY_PH_URBAN_FACADE_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_realistic/polyhaven/usd/modular_urban_apartments_facade_1k.usd"
CITY_PH_FACTORY_FACADE_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_realistic/polyhaven/usd/modular_factory_facade_1k.usd"
CITY_SOURCECITY_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_complete/sourcecity_usda_fixed/Main_Intersection_v2.usda"
CITY_NVIDIA_DEMO_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Assemblies/assembly_City.usd"
CITY_SHORT_ASPHALT_ROAD_USD_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_realistic/road/short_asphalt_road.usda"
CITY_NVIDIA_ASPHALT_MDL_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Source/context_City/rh_Context_City/Materials/Asphalt/Asphalt.mdl"
CITY_NVIDIA_CONCRETE_MDL_PATH = "/home/cnc/SSD_1T/xzw/IsaacLab-main/assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Source/context_City/rh_Context_City/Materials/Concrete_SmoothPanels_uk2iadmew.mdl"
CITY_BUILDING_NORTH_ROT = (1.0, 0.0, 0.0, 0.0)
CITY_BUILDING_SOUTH_ROT = (0.0, 0.0, 0.0, 1.0)
CITY_REAL_NORTH_ROT = (0.7071068, 0.7071068, 0.0, 0.0)
CITY_REAL_SOUTH_ROT = (0.7071068, -0.7071068, 0.0, 0.0)
CITY_FACADE_NORTH_ROT = (0.7071068, 0.7071068, 0.0, 0.0)
CITY_FACADE_SOUTH_ROT = (0.7071068, -0.7071068, 0.0, 0.0)
CITY_NEAR_BUILDING_SCALE = (1.18, 1.18, 1.18)
CITY_BACK_BUILDING_SCALE = (0.82, 0.82, 0.82)
CITY_REAL_BUILDING_SCALE = (0.030, 0.030, 0.030)
CITY_REAL_STATION_SCALE = (0.012, 0.012, 0.012)
CITY_URBAN_STREET_FACADE_SCALE = (0.170, 0.315, 0.052)
CITY_FACTORY_STREET_FACADE_SCALE = (0.158, 0.250, 0.052)
CITY_URBAN_BACK_FACADE_SCALE = (0.145, 0.270, 0.045)
CITY_FACTORY_BACK_FACADE_SCALE = (0.135, 0.220, 0.045)
CITY_STREET_FACADE_Z = -20.0
CITY_BACK_FACADE_Z = -20.0
CITY_SOURCECITY_SCALE = (0.045, 0.045, 0.045)
CITY_NVIDIA_DEMO_SCALE = (0.0045, 0.0045, 0.0045)
CITY_NVIDIA_DEMO_ROT_Z90 = (0.7071068, 0.0, 0.0, 0.7071068)
CITY_LANE_CENTER_Y = 0.0
CITY_OTHER_LANE_CENTER_Y = -1.20
CITY_LANE_DIVIDER_Y = -0.60
TARGET_SPEED_RANGE = (0.6, 0.8)
WHEEL_ACTION_SCALE = 12.0
WHEEL_VELOCITY_LIMIT = 20.0

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
            stiffness=0.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=WHEEL_VELOCITY_LIMIT,
        ),
    }
)

PLATOON_ROBOTS = ["robot", "robot_2", "robot_3", "robot_4", "robot_5"]
CTH_STANDSTILL_DISTANCE = 1.5
CTH_HEADWAY = 0.6
LATERAL_DEADZONE = 0.05
LATERAL_CRITICAL_OFFSET = 0.25
LATERAL_MAX_PAIR_WEIGHT = 0.2
LATERAL_PAIR2_EXTRA_WEIGHT = 0.0
LATERAL_PAIR2_INDEX = 1  # pair_2: robot_2 -> robot_3
CENTERLINE_Y_TARGET = 0.0
CENTERLINE_DEADZONE = 0.05
CENTERLINE_CRITICAL_OFFSET = 0.35
CENTERLINE_MAX_WEIGHT = 0.5


def _cth_desired_gap(env, follower_name: str):
    follower_speed = torch.clamp(env.scene[follower_name].data.root_lin_vel_b[:, 0], min=0.0)
    return CTH_STANDSTILL_DISTANCE + CTH_HEADWAY * follower_speed


def _env_local_y(env, asset_name: str):
    pos_y = env.scene[asset_name].data.root_pos_w[:, 1]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        pos_y = pos_y - env_origins[:, 1].to(device=pos_y.device, dtype=pos_y.dtype)
    return pos_y


def _city_box(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.75,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"/World/CityScenery/{name}",
        spawn=CuboidCfg(
            size=size,
            visual_material=PreviewSurfaceCfg(
                diffuse_color=color,
                emissive_color=emissive,
                metallic=metallic,
                roughness=roughness,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _city_box_mdl(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    mdl_path: str,
    *,
    texture_scale: tuple[float, float] | None = None,
    albedo_brightness: float | None = None,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"/World/CityScenery/{name}",
        spawn=CuboidCfg(
            size=size,
            visual_material=MdlFileCfg(
                mdl_path=mdl_path,
                project_uvw=True,
                texture_scale=texture_scale,
                albedo_brightness=albedo_brightness,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _city_cylinder(
    name: str,
    pos: tuple[float, float, float],
    radius: float,
    height: float,
    color: tuple[float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.55,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"/World/CityScenery/{name}",
        spawn=CylinderCfg(
            radius=radius,
            height=height,
            visual_material=PreviewSurfaceCfg(
                diffuse_color=color,
                emissive_color=emissive,
                metallic=metallic,
                roughness=roughness,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _city_usd(
    name: str,
    usd_path: str,
    pos: tuple[float, float, float],
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"/World/CityScenery/{name}",
        spawn=UsdFileCfg(usd_path=usd_path, scale=scale),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
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
    robots = PLATOON_ROBOTS
    tolerance = 0.2  # 苛刻的成功标准：10cm 误差

    all_success_mask = torch.ones(env.num_envs, device=env.device)

    for i in range(1, len(robots)):
        dist = torch.norm(env.scene[robots[i - 1]].data.root_pos_w[:, :2] -
                          env.scene[robots[i]].data.root_pos_w[:, :2], dim=1)
        target_dist = _cth_desired_gap(env, robots[i])
        # 只要有一对车距离不达标，这一帧就算失败 (0分)
        is_good = (torch.abs(dist - target_dist) < tolerance).float()
        all_success_mask *= is_good

    return all_success_mask


def reward_true_collision_fail(env):
    """
    [论文 \bar{R}] 绝对失败指标 (Sparse Reward)
    只要发生任何碰撞 (距离 < 30cm)，就算严重事故。
    """
    robots = PLATOON_ROBOTS
    min_safe_dist = 0.30
    any_crash_mask = torch.zeros(env.num_envs, device=env.device)

    for i in range(1, len(robots)):
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

    debug_step = int(getattr(env, "common_step_counter", 0))
    if debug_step >= 20 and debug_step % 60 == 20:
        printed_steps = getattr(env, "_debug_leader_vel_steps", set())
        if debug_step in printed_steps:
            return reward
        printed_steps.add(debug_step)
        env._debug_leader_vel_steps = printed_steps
        action_terms = getattr(getattr(env, "action_manager", None), "_terms", {})
        leader_action = action_terms.get("joint_vel_1") if isinstance(action_terms, dict) else None
        raw_action = getattr(leader_action, "raw_actions", None)
        processed_action = getattr(leader_action, "processed_actions", None)
        joint_vel = env.scene["robot"].data.joint_vel
        print("[DEBUG][leader_motion] step=", debug_step)
        print("[DEBUG][leader_motion] command_vel[0]=", command_vel[0].detach().cpu().tolist())
        print("[DEBUG][leader_motion] actual_vel[0]=", actual_vel[0].detach().cpu().tolist())
        if raw_action is not None:
            print("[DEBUG][leader_motion] raw_action[0]=", raw_action[0].detach().cpu().tolist())
        if processed_action is not None:
            print("[DEBUG][leader_motion] processed_action[0]=", processed_action[0].detach().cpu().tolist())
        print("[DEBUG][leader_motion] wheel_joint_vel[0]=", joint_vel[0].detach().cpu().tolist())
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
    robots = PLATOON_ROBOTS
    total_score = 0
    # CTH spacing: d_des = d0 + h * v_i.  Here d0 is scaled to the
    # small-car/URDF experiment distance while preserving the CTH structure.

    # 1. 动态 Sigma (保持之前的逻辑，用于从宽容变严厉)
    current_step = env.common_step_counter
    # 注意：这里的时间表可以根据你的实际步数调整，之前是50M，现在如果是Play模式其实只看最终效果
    # 训练时建议：start=2.0, end=0.2
    # Play时其实这个函数不起作用（Play只跑模型，不训练），但为了逻辑一致保持原样
    progress = min(current_step / 50_000_000.0, 1.0)
    current_sigma = 2.0 + (0.2 - 2.0) * progress

    for i in range(1, len(robots)):
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
        desired_gap = _cth_desired_gap(env, robots[i])
        error_longitudinal = rel_x - (-desired_gap)

        # 2. 横向误差：我们希望后车在正后方，不能偏左偏右
        # 所以 rel_y 应该是 0.0
        error_lateral = rel_y - 0.0

        # 3. 综合评分
        # 我们对横向误差惩罚得更重一点 (sigma/2)，强迫它们走直线
        score_long = torch.exp(-torch.square(error_longitudinal) / current_sigma)
        score_lat = torch.exp(-torch.square(error_lateral) / (current_sigma / 2.0))

        # 两个维度都好，才给高分
        total_score += score_long * score_lat

    return total_score / float(len(robots) - 1)


def reward_lateral_penalty_hard(env):
    """
    Penalize follower lateral offset from its predecessor's centerline.

    The penalty is intentionally asymmetric in severity: tiny offsets inside a
    5 cm dead-zone are ignored, but visible lane drift grows faster after
    25 cm. The aggregation includes the worst pair so a single bad follower
    relation cannot be hidden by a low platoon mean.
    """
    robots = PLATOON_ROBOTS
    pair_penalties = []

    for i in range(1, len(robots)):
        pos_leader = env.scene[robots[i - 1]].data.root_pos_w[:, :2]
        pos_follower = env.scene[robots[i]].data.root_pos_w[:, :2]
        quat_leader = env.scene[robots[i - 1]].data.root_quat_w

        diff_vec = pos_follower - pos_leader
        rel_pos = quat_apply_inverse(quat_leader, torch.cat([diff_vec, torch.zeros_like(diff_vec[:, 0:1])], dim=1))

        abs_lat = torch.abs(rel_pos[:, 1])
        soft_error = torch.relu(abs_lat - LATERAL_DEADZONE)
        hard_error = torch.relu(abs_lat - LATERAL_CRITICAL_OFFSET)
        pair_penalties.append(soft_error + 4.0 * torch.square(hard_error))

    if not pair_penalties:
        return torch.zeros(env.num_envs, device=env.device)

    pair_penalty = torch.stack(pair_penalties, dim=1)
    mean_penalty = pair_penalty.mean(dim=1)
    max_penalty = pair_penalty.max(dim=1).values
    pair2_penalty = pair_penalty[:, min(LATERAL_PAIR2_INDEX, pair_penalty.shape[1] - 1)]
    return mean_penalty + LATERAL_MAX_PAIR_WEIGHT * max_penalty + LATERAL_PAIR2_EXTRA_WEIGHT * pair2_penalty


def reward_lateral_velocity_penalty(env):
    """Penalize side-slip and lateral velocity mismatch in the platoon."""
    robots = PLATOON_ROBOTS
    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(1, len(robots)):
        pred = env.scene[robots[i - 1]]
        foll = env.scene[robots[i]]
        pred_vy = pred.data.root_lin_vel_b[:, 1]
        foll_vy = foll.data.root_lin_vel_b[:, 1]
        own_side_slip = torch.abs(foll_vy)
        rel_side_slip = torch.abs(foll_vy - pred_vy)
        total_penalty += own_side_slip + 0.5 * rel_side_slip
    return total_penalty / float(len(robots) - 1)


def reward_centerline_lateral_penalty(env):
    """Penalize env-local vehicle offset from the road centerline y=0."""
    robot_penalties = []
    for name in PLATOON_ROBOTS:
        pos_y = _env_local_y(env, name)
        abs_y = torch.abs(pos_y - CENTERLINE_Y_TARGET)
        soft_error = torch.relu(abs_y - CENTERLINE_DEADZONE)
        hard_error = torch.relu(abs_y - CENTERLINE_CRITICAL_OFFSET)
        robot_penalties.append(soft_error + 3.0 * torch.square(hard_error))

    if not robot_penalties:
        return torch.zeros(env.num_envs, device=env.device)

    centerline_penalty = torch.stack(robot_penalties, dim=1)
    return centerline_penalty.mean(dim=1) + CENTERLINE_MAX_WEIGHT * centerline_penalty.max(dim=1).values


def reward_heading_alignment_penalty(env):
    """Penalize yaw/heading disagreement that lets followers peel sideways."""
    robots = PLATOON_ROBOTS
    target_dir = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    headings = []
    for name in robots:
        heading = quat_apply(env.scene[name].data.root_quat_w, target_dir)
        headings.append(torch.nn.functional.normalize(heading[:, :2], dim=-1))

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(headings)):
        world_alignment = torch.clamp(headings[i][:, 0], -1.0, 1.0)
        total_penalty += 1.0 - world_alignment
        if i > 0:
            pair_alignment = torch.clamp(torch.sum(headings[i - 1] * headings[i], dim=-1), -1.0, 1.0)
            total_penalty += 0.5 * (1.0 - pair_alignment)
    return total_penalty / float(len(robots))

def reward_collision_risk(env):
    """[Dense Reward] 碰撞风险梯度惩罚"""
    robots = PLATOON_ROBOTS
    penalty = 0
    min_safe_dist = 0.35
    for i in range(1, len(robots)):
        dist = torch.norm(env.scene[robots[i - 1]].data.root_pos_w[:, :2] -
                          env.scene[robots[i]].data.root_pos_w[:, :2], dim=1)
        is_too_close = (dist < min_safe_dist).float()
        penalty += is_too_close
    return penalty


def reward_move_backward_penalty(env):
    """禁止倒车"""
    penalty = 0
    robots = PLATOON_ROBOTS
    for name in robots:
        vel_x = env.scene[name].data.root_lin_vel_b[:, 0]
        is_backing = (vel_x < -0.05).float()
        penalty += is_backing
    return penalty


def reward_forward_drive_chain(env):
    """Reward normalized forward speed for the full platoon."""
    command_vel = env.command_manager.get_command("base_velocity")
    target_speed = torch.clamp(command_vel[:, 0].abs(), min=0.1)
    total = torch.zeros(env.num_envs, device=env.device)
    robots = PLATOON_ROBOTS
    for name in robots:
        vel_x = env.scene[name].data.root_lin_vel_b[:, 0]
        total += torch.clamp(vel_x, min=0.0) / target_speed
    return total / float(len(robots))


def reward_leader_progress(env):
    """Reward actual leader x displacement, normalized by commanded progress."""
    leader_x = env.scene["robot"].data.root_pos_w[:, 0]
    if not hasattr(env, "_platoon_prev_leader_x"):
        env._platoon_prev_leader_x = leader_x.detach().clone()
        return torch.zeros(env.num_envs, device=env.device)

    prev_x = env._platoon_prev_leader_x.to(device=env.device)
    delta_x = leader_x - prev_x
    env._platoon_prev_leader_x = leader_x.detach().clone()

    step_dt = float(getattr(env, "step_dt", 0.02))
    command_vel = env.command_manager.get_command("base_velocity")
    expected_delta = torch.clamp(command_vel[:, 0].abs() * step_dt, min=1.0e-4)
    progress = torch.clamp(delta_x, min=0.0) / expected_delta

    just_reset = env.episode_length_buf <= 1
    large_jump = torch.abs(delta_x) > 2.0
    return torch.where(just_reset | large_jump, torch.zeros_like(progress), torch.clamp(progress, max=1.5))


def reward_stall_penalty(env):
    """Penalize standing still when the command asks the platoon to move forward."""
    command_vel = env.command_manager.get_command("base_velocity")
    target_speed = torch.clamp(command_vel[:, 0], min=0.0)
    min_speed = torch.clamp(0.35 * target_speed, min=0.12)
    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in PLATOON_ROBOTS:
        vel_x = env.scene[name].data.root_lin_vel_b[:, 0]
        penalty += ((target_speed > 0.1) & (vel_x < min_speed)).float()
    return penalty / float(len(PLATOON_ROBOTS))


def termination_bad_heading_chain(env):
    """检查掉头 (>90度)"""
    robots = PLATOON_ROBOTS
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
    """Per-agent local observation flattened for IsaacLab.

    Layout per agent:
    [
        rel_x, rel_y, rel_vx, rel_vy,
        own_vx, own_vy, own_yaw_rate,
        centerline_y, heading_y, pair_heading_sin,
        role_id,
    ].
    The leader receives zeros for predecessor-relative terms.
    """
    robots = PLATOON_ROBOTS
    obs_list = []
    target_dir = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    heading_vecs = []
    for name in robots:
        heading = quat_apply(env.scene[name].data.root_quat_w, target_dir)
        heading_vecs.append(torch.nn.functional.normalize(heading[:, :2], dim=-1))

    for i, name in enumerate(robots):
        car = env.scene[name]
        own_vel = car.data.root_lin_vel_b[:, :2]
        own_yaw_rate = car.data.root_ang_vel_b[:, 2:3]
        centerline_y = torch.clamp((_env_local_y(env, name) - CENTERLINE_Y_TARGET).unsqueeze(-1), -2.0, 2.0)
        heading_y = heading_vecs[i][:, 1:2]
        role_id = torch.full((env.num_envs, 1), float(i) / float(max(len(robots) - 1, 1)), device=env.device)
        if i == 0:
            rel_terms = torch.zeros(env.num_envs, 4, device=env.device)
            pair_heading_sin = torch.zeros(env.num_envs, 1, device=env.device)
        else:
            pred = env.scene[robots[i - 1]]
            rel_pos_local = quat_apply_inverse(car.data.root_quat_w, pred.data.root_pos_w - car.data.root_pos_w)
            rel_vel = pred.data.root_lin_vel_b[:, :2] - car.data.root_lin_vel_b[:, :2]
            rel_terms = torch.cat([rel_pos_local[:, :2], rel_vel], dim=-1)
            pred_heading = heading_vecs[i - 1]
            own_heading = heading_vecs[i]
            pair_heading_sin = (
                pred_heading[:, 0:1] * own_heading[:, 1:2]
                - pred_heading[:, 1:2] * own_heading[:, 0:1]
            )
        obs_list.append(
            torch.cat(
                [rel_terms, own_vel, own_yaw_rate, centerline_y, heading_y, pair_heading_sin, role_id],
                dim=-1,
            )
        )
    return torch.cat(obs_list, dim=-1)


# =============================================================================
# 3. 配置组定义
# =============================================================================

@configclass
class PlatoonObservationsCfg:
    @configclass
    class PolicyGroupCfg(ObservationGroupCfg):
        rel_pos_chain = ObservationTermCfg(func=obs_platoon_chain)

    policy: PolicyGroupCfg = PolicyGroupCfg()


@configclass
class PlatoonRewardsCfg:
    # --- Inner Loop: 训练引导信号 (Dense) ---
    # Progress-aware reward: make actual forward movement dominate survival.
    leader_motion = RewardTermCfg(func=reward_leader_velocity_matching, weight=5.0)
    leader_progress = RewardTermCfg(func=reward_leader_progress, weight=6.0)
    formation = RewardTermCfg(func=reward_platoon_dist_chain_dynamic, weight=2.0)
    collision_risk = RewardTermCfg(func=reward_collision_risk, weight=-12.0)
    no_backward = RewardTermCfg(func=reward_move_backward_penalty, weight=-10.0)
    forward_drive = RewardTermCfg(func=reward_forward_drive_chain, weight=6.0)
    stall_penalty = RewardTermCfg(func=reward_stall_penalty, weight=-4.0)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.10)
    lateral_correct = RewardTermCfg(func=reward_lateral_penalty_hard, weight=-3.0)
    lateral_velocity = RewardTermCfg(func=reward_lateral_velocity_penalty, weight=-0.8)
    centerline_lateral = RewardTermCfg(func=reward_centerline_lateral_penalty, weight=-0.3)
    heading_align = RewardTermCfg(func=reward_heading_alignment_penalty, weight=-0.8)

    # --- Outer Loop: 真实目标评估 (Sparse) ---
    # 我们给一个非零的小权重，这样它们会显示在 Tensorboard 日志里
    # 你可以通过这两个指标来判断你的 Dense Reward 设计得好不好
    true_success = RewardTermCfg(func=reward_true_formation_success, weight=0.5)
    true_fail = RewardTermCfg(func=reward_true_collision_fail, weight=-1.0)

    alive = RewardTermCfg(func=mdp.is_alive, weight=0.2)


@configclass
class PlatoonEnvCfg(ManagerBasedRLEnvCfg):
    @configclass
    class PlatoonAlgorithmCfg:
        """Task-local algorithm mode flags.

        PPO remains the default stable baseline. HAPPO/teacher/attack/shield are
        toggles for the task internals, not separate launch entrypoints.
        """

        algorithm: str = "ppo"
        num_agents: int = 5
        enable_teacher: bool = False
        enable_attack: bool = False
        enable_shield: bool = False
        use_happo_layout: bool = False
        use_happo_actions: bool = False
        happo_log_level: str = "basic"
        happo_action_clip: float = 0.4
        happo_action_warmup_updates: int = 0
        happo_urdf_wheel_sign_adapter: bool = True
        happo_entropy_coef: float = 0.0
        happo_init_noise_std: float = 0.25
        happo_log_std_min: float = -2.995732273553991  # log(0.05)
        happo_log_std_max: float = -1.0498221244986778  # log(0.35)
        local_reward_shaping: bool = True
        local_reward_centerline_coef: float = 0.35
        local_reward_pair_lateral_coef: float = 0.55
        local_reward_gap_coef: float = 0.0
        local_reward_heading_coef: float = 0.15
        local_reward_turn_coef: float = 0.005
        local_reward_first_follower_centerline_scale: float = 1.0
        local_reward_first_follower_pair_lateral_scale: float = 1.0
        local_reward_first_follower_gap_scale: float = 1.0
        local_reward_first_follower_turn_scale: float = 1.0
        local_reward_last_follower_centerline_scale: float = 1.0
        local_reward_last_follower_pair_lateral_scale: float = 1.0
        local_reward_last_follower_gap_scale: float = 1.0
        local_reward_last_follower_turn_scale: float = 1.0
        teacher_shaping_coef: float = 0.02
        teacher_lr: float = 3.0e-4
        teacher_update_interval: int = 1
        teacher_every_student_updates: int = 2
        teacher_shaping_clip: float = 0.20
        teacher_action_penalty_coef: float = 0.001
        teacher_reward_ema_tau: float = 0.95
        teacher_consistency_coef: float = 0.03
        teacher_outer_delta_coef: float = 0.08
        teacher_outer_delta_warmup_updates: int = 5
        teacher_outer_delta_ramp_updates: int = 20
        teacher_lambda_spacing: float = 1.0
        teacher_lambda_velocity: float = 0.5
        teacher_lambda_acceleration: float = 0.25
        teacher_lambda_jerk: float = 0.10
        teacher_lambda_overspeed: float = 0.25
        teacher_lambda_centerline: float = 1.0
        teacher_lambda_lateral: float = 0.5
        teacher_lambda_heading: float = 2.0
        teacher_lambda_backward: float = 1.0
        teacher_lambda_forward_deficit: float = 0.5
        teacher_lambda_collision: float = 10.0
        teacher_lambda_action_energy: float = 0.02
        max_fdi_pos: float = -1.0
        max_fdi_acc: float = -1.0
        max_dos_rate: float = -1.0

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        # --- Robot 1 (领航车): 深海蓝 ---
        robot = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        robot.init_state.pos = (-8.75, 0.0, 0.80)
        robot.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.10, 0.42),
            metallic=0.55, roughness=0.28
        )

        # --- Robot 2 (跟随车): 珍珠白，原 Robot 3 顺移 ---
        robot_2 = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        robot_2.init_state.pos = (-10.25, 0.0, 0.80)
        robot_2.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.86, 0.86, 0.80),
            metallic=0.18, roughness=0.32
        )

        # --- Robot 3 (跟随车): 标准红，原 Robot 4 顺移 ---
        robot_3 = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3")
        robot_3.init_state.pos = (-11.75, 0.0, 0.80)
        robot_3.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.62, 0.03, 0.025),
            metallic=0.42, roughness=0.30
        )

        # --- Robot 4 (跟随车): 金属银，原 Robot 5 顺移 ---
        robot_4 = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_4")
        robot_4.init_state.pos = (-13.25, 0.0, 0.80)
        robot_4.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.58, 0.61, 0.62),
            metallic=0.65, roughness=0.24
        )

        # --- Robot 5 (跟随车): 石墨黑，原 Robot 6 顺移 ---
        robot_5 = MY_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_5")
        robot_5.init_state.pos = (-14.75, 0.0, 0.80)
        robot_5.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.035, 0.040, 0.045),
            metallic=0.55, roughness=0.30
        )

        # 2. 地形配置 (本地程序化平面，避免阶段切换时依赖远程 Omniverse grid USD)
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(2000.0, 2000.0),
                num_rows=1,
                num_cols=1,
                border_width=0.0,
                color_scheme="none",
                use_cache=False,
                sub_terrains={
                    "flat": MeshPlaneTerrainCfg(proportion=1.0),
                },
            ),
            use_terrain_origins=False,
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.89), roughness=0.86),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        # 3. 高速公路视觉资产
        highway_scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/HighwayScene",
            spawn=UsdFileCfg(usd_path=HIGHWAY_USD_PATH),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -20.0))
        )

        # 4. 高速公路物理路面
        highway_collision = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/HighwayCollision",
            spawn=CuboidCfg(
                visible=False,
                size=(2000.0, 2.95, 0.1),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.028, 0.031, 0.034)),
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
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.55, 0.082))
        )

        # Global visual-only city dressing for single-env play/rendering.
        # y=0 is intentionally the near-building lane center used by the policy.
        city_full_map = _city_usd(
            "nvidia_city_demo",
            CITY_NVIDIA_DEMO_USD_PATH,
            (10.0, -33.0, -0.12),
            scale=CITY_NVIDIA_DEMO_SCALE,
            rot=CITY_NVIDIA_DEMO_ROT_Z90,
        )
        city_road_asphalt = _city_usd("two_lane_asphalt", CITY_SHORT_ASPHALT_ROAD_USD_PATH, (2.0, -0.54, 0.010))
        city_road_near_lane = _city_box("near_lane_subtle_wear", (0.0, CITY_LANE_CENTER_Y, -20.0), (1.0, 1.0, 0.006), (0.042, 0.045, 0.047), roughness=0.96)
        city_road_far_lane = _city_box("far_lane_subtle_wear", (0.0, CITY_OTHER_LANE_CENTER_Y, -20.0), (1.0, 1.0, 0.006), (0.036, 0.038, 0.040), roughness=0.96)
        city_curb_near = _city_box_mdl("curb_near_buildings", (2.0, 0.66, 0.050), (52.0, 0.080, 0.026), CITY_NVIDIA_CONCRETE_MDL_PATH, texture_scale=(0.45, 0.45), albedo_brightness=0.72)
        city_curb_far = _city_box_mdl("curb_far_side", (2.0, -1.93, 0.050), (52.0, 0.070, 0.022), CITY_NVIDIA_CONCRETE_MDL_PATH, texture_scale=(0.45, 0.45), albedo_brightness=0.65)
        city_sidewalk_near = _city_box_mdl("sidewalk_near_buildings", (2.0, 0.98, 0.044), (52.0, 0.62, 0.012), CITY_NVIDIA_CONCRETE_MDL_PATH, texture_scale=(0.35, 0.35), albedo_brightness=0.60)
        city_near_edge_paving = _city_box("near_edge_paving", (2.0, 0.50, 0.096), (52.0, 0.34, 0.012), (0.78, 0.78, 0.74), roughness=0.88)
        city_sidewalk_far = _city_box("sidewalk_far_side", (2.0, -3.05, 0.042), (52.0, 2.05, 0.010), (0.86, 0.86, 0.83), roughness=0.88)
        city_far_edge_paving = _city_box("far_edge_paving", (2.0, -2.12, 0.094), (52.0, 0.78, 0.012), (0.90, 0.90, 0.87), roughness=0.88)
        city_clean_outer_ground = _city_box("clean_outer_ground", (2.0, -6.25, 0.034), (52.0, 4.30, 0.010), (0.90, 0.90, 0.87), roughness=0.88)
        city_urban_forecourt = _city_box_mdl("urban_forecourt", (2.0, 2.70, 0.043), (52.0, 3.15, 0.012), CITY_NVIDIA_CONCRETE_MDL_PATH, texture_scale=(0.28, 0.28), albedo_brightness=0.55)
        city_urban_forecourt_strip_01 = _city_box("urban_forecourt_strip_01", (2.0, 1.30, 0.052), (52.0, 0.036, 0.006), (0.40, 0.41, 0.40), roughness=0.90)
        city_urban_forecourt_strip_02 = _city_box("urban_forecourt_strip_02", (2.0, 4.15, 0.052), (52.0, 0.036, 0.006), (0.32, 0.33, 0.33), roughness=0.90)
        city_edge_line_near = _city_box("edge_line_near", (2.0, 0.47, 0.016), (52.0, 0.026, 0.003), (0.82, 0.82, 0.76))
        city_edge_line_far = _city_box("edge_line_far", (2.0, -1.72, 0.016), (52.0, 0.026, 0.003), (0.82, 0.82, 0.76))

        city_intersection_west = _city_box("intersection_west_asphalt", (-62.0, -0.60, 0.052), (4.40, 5.30, 0.008), (0.026, 0.028, 0.030), roughness=0.94)
        city_intersection_mid = _city_box("intersection_mid_asphalt", (-12.0, -0.60, -20.0), (4.00, 5.30, 0.008), (0.026, 0.028, 0.030), roughness=0.94)
        city_intersection_east = _city_box("intersection_east_asphalt", (58.0, -0.60, 0.052), (4.40, 5.30, 0.008), (0.026, 0.028, 0.030), roughness=0.94)
        city_stop_bar_west_near = _city_box("stop_bar_west_near_lane", (-59.55, CITY_LANE_CENTER_Y, 0.070), (0.12, 0.86, 0.010), (0.88, 0.88, 0.82))
        city_stop_bar_west_far = _city_box("stop_bar_west_far_lane", (-59.55, CITY_OTHER_LANE_CENTER_Y, 0.070), (0.12, 0.86, 0.010), (0.88, 0.88, 0.82))
        city_crosswalk_mid_01 = _city_box("crosswalk_mid_01", (-12.75, -0.60, -20.0), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_mid_02 = _city_box("crosswalk_mid_02", (-12.30, -0.60, -20.0), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_mid_03 = _city_box("crosswalk_mid_03", (-11.85, -0.60, -20.0), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_mid_04 = _city_box("crosswalk_mid_04", (-11.40, -0.60, -20.0), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_stop_bar_mid_near = _city_box("stop_bar_mid_near_lane", (-9.60, CITY_LANE_CENTER_Y, -20.0), (0.12, 0.86, 0.010), (0.88, 0.88, 0.82))
        city_stop_bar_mid_far = _city_box("stop_bar_mid_far_lane", (-9.60, CITY_OTHER_LANE_CENTER_Y, -20.0), (0.12, 0.86, 0.010), (0.88, 0.88, 0.82))
        city_stop_bar_east_near = _city_box("stop_bar_east_near_lane", (55.55, CITY_LANE_CENTER_Y, 0.070), (0.12, 0.86, 0.010), (0.88, 0.88, 0.82))
        city_stop_bar_east_far = _city_box("stop_bar_east_far_lane", (55.55, CITY_OTHER_LANE_CENTER_Y, 0.070), (0.12, 0.86, 0.010), (0.88, 0.88, 0.82))

        city_lane_dash_01 = _city_box("lane_divider_dash_01", (-54.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_02 = _city_box("lane_divider_dash_02", (-45.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_03 = _city_box("lane_divider_dash_03", (-36.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_04 = _city_box("lane_divider_dash_04", (-27.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_05 = _city_box("lane_divider_dash_05", (-18.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_06 = _city_box("lane_divider_dash_06", (-9.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_07 = _city_box("lane_divider_dash_07", (0.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_08 = _city_box("lane_divider_dash_08", (9.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_09 = _city_box("lane_divider_dash_09", (18.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_10 = _city_box("lane_divider_dash_10", (27.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_11 = _city_box("lane_divider_dash_11", (36.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_12 = _city_box("lane_divider_dash_12", (45.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))
        city_lane_dash_13 = _city_box("lane_divider_dash_13", (54.0, CITY_LANE_DIVIDER_Y, 0.016), (1.45, 0.038, 0.003), (0.86, 0.65, 0.16))

        city_crosswalk_a01 = _city_box("crosswalk_a01", (-62.0, -0.60, 0.068), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_a02 = _city_box("crosswalk_a02", (-61.55, -0.60, 0.068), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_a03 = _city_box("crosswalk_a03", (-61.10, -0.60, 0.068), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_b01 = _city_box("crosswalk_b01", (58.0, -0.60, 0.068), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_b02 = _city_box("crosswalk_b02", (58.45, -0.60, 0.068), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))
        city_crosswalk_b03 = _city_box("crosswalk_b03", (58.90, -0.60, 0.068), (0.16, 2.10, 0.010), (0.86, 0.86, 0.80))

        city_block_body_n01 = _city_box("city_block_body_n01", (-151.0, 3.02, 1.34), (7.2, 1.18, 2.68), (0.55, 0.50, 0.44), roughness=0.84)
        city_block_body_n02 = _city_box("city_block_body_n02", (-139.0, 3.10, 1.56), (8.4, 1.30, 3.12), (0.36, 0.39, 0.42), roughness=0.80)
        city_block_body_n03 = _city_box("city_block_body_n03", (-126.5, 3.06, 1.48), (7.6, 1.22, 2.96), (0.48, 0.43, 0.39), roughness=0.86)
        city_block_body_n04 = _city_box("city_block_body_n04", (70.0, 3.42, 1.26), (6.8, 1.16, 2.52), (0.42, 0.44, 0.40), roughness=0.82)
        city_block_body_n05 = _city_box("city_block_body_n05", (83.0, 3.52, 1.68), (8.6, 1.34, 3.36), (0.32, 0.36, 0.40), roughness=0.78)
        city_block_body_n06 = _city_box("city_block_body_n06", (96.0, 3.04, 1.44), (7.4, 1.20, 2.88), (0.52, 0.46, 0.40), roughness=0.84)
        city_block_body_n07 = _city_box("city_block_body_n07", (110.0, 3.02, 1.30), (7.2, 1.18, 2.60), (0.40, 0.42, 0.39), roughness=0.82)
        city_block_body_n08 = _city_box("city_block_body_n08", (123.0, 3.10, 1.60), (8.2, 1.30, 3.20), (0.34, 0.38, 0.42), roughness=0.80)
        city_block_body_n09 = _city_box("city_block_body_n09", (136.0, 3.06, 1.48), (7.6, 1.22, 2.96), (0.54, 0.48, 0.42), roughness=0.84)
        city_block_roof_n01 = _city_box("city_block_roof_n01", (-151.0, 3.02, 2.72), (7.6, 1.34, 0.12), (0.16, 0.17, 0.17), metallic=0.04, roughness=0.70)
        city_block_roof_n02 = _city_box("city_block_roof_n02", (-139.0, 3.10, 3.17), (8.8, 1.46, 0.12), (0.13, 0.14, 0.15), metallic=0.04, roughness=0.70)
        city_block_roof_n03 = _city_box("city_block_roof_n03", (-126.5, 3.06, 3.00), (8.0, 1.38, 0.12), (0.17, 0.15, 0.14), metallic=0.04, roughness=0.70)
        city_block_roof_n04 = _city_box("city_block_roof_n04", (70.0, 3.42, 2.58), (7.2, 1.32, 0.12), (0.14, 0.15, 0.15), metallic=0.04, roughness=0.70)
        city_block_roof_n05 = _city_box("city_block_roof_n05", (83.0, 3.52, 3.41), (9.0, 1.50, 0.12), (0.12, 0.13, 0.14), metallic=0.04, roughness=0.70)
        city_block_roof_n06 = _city_box("city_block_roof_n06", (96.0, 3.04, 2.92), (7.8, 1.36, 0.12), (0.18, 0.16, 0.14), metallic=0.04, roughness=0.70)
        city_block_roof_n07 = _city_box("city_block_roof_n07", (110.0, 3.02, 2.64), (7.6, 1.34, 0.12), (0.14, 0.15, 0.14), metallic=0.04, roughness=0.70)
        city_block_roof_n08 = _city_box("city_block_roof_n08", (123.0, 3.10, 3.24), (8.6, 1.46, 0.12), (0.12, 0.14, 0.16), metallic=0.04, roughness=0.70)
        city_block_roof_n09 = _city_box("city_block_roof_n09", (136.0, 3.06, 3.00), (8.0, 1.38, 0.12), (0.18, 0.16, 0.14), metallic=0.04, roughness=0.70)
        city_block_base_n01 = _city_box("city_block_base_n01", (-151.0, 2.32, 0.20), (7.4, 0.20, 0.40), (0.22, 0.22, 0.21), roughness=0.82)
        city_block_base_n02 = _city_box("city_block_base_n02", (-139.0, 2.34, 0.22), (8.4, 0.20, 0.44), (0.20, 0.21, 0.22), roughness=0.82)
        city_block_base_n03 = _city_box("city_block_base_n03", (-126.5, 2.32, 0.20), (7.7, 0.20, 0.40), (0.23, 0.22, 0.20), roughness=0.82)
        city_block_base_n04 = _city_box("city_block_base_n04", (70.0, 2.72, 0.18), (6.9, 0.20, 0.36), (0.20, 0.22, 0.20), roughness=0.82)
        city_block_base_n05 = _city_box("city_block_base_n05", (83.0, 2.74, 0.22), (8.6, 0.20, 0.44), (0.18, 0.20, 0.22), roughness=0.82)
        city_block_base_n06 = _city_box("city_block_base_n06", (96.0, 2.32, 0.20), (7.5, 0.20, 0.40), (0.22, 0.21, 0.19), roughness=0.82)
        city_block_base_n07 = _city_box("city_block_base_n07", (110.0, 2.32, 0.18), (7.2, 0.20, 0.36), (0.20, 0.22, 0.20), roughness=0.82)
        city_block_base_n08 = _city_box("city_block_base_n08", (123.0, 2.34, 0.22), (8.3, 0.20, 0.44), (0.18, 0.20, 0.22), roughness=0.82)
        city_block_base_n09 = _city_box("city_block_base_n09", (136.0, 2.32, 0.20), (7.7, 0.20, 0.40), (0.22, 0.21, 0.19), roughness=0.82)

        city_facade_n01 = _city_usd("urban_facade_n01", CITY_PH_URBAN_FACADE_USD_PATH, (-51.0, 2.20, CITY_STREET_FACADE_Z), scale=CITY_URBAN_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n02 = _city_usd("factory_facade_n02", CITY_PH_FACTORY_FACADE_USD_PATH, (-39.0, 2.20, CITY_STREET_FACADE_Z), scale=CITY_FACTORY_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n03 = _city_usd("urban_facade_n03", CITY_PH_URBAN_FACADE_USD_PATH, (-26.5, 2.20, CITY_STREET_FACADE_Z), scale=CITY_URBAN_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n04 = _city_usd("factory_facade_n04", CITY_PH_FACTORY_FACADE_USD_PATH, (70.0, 2.60, CITY_STREET_FACADE_Z), scale=CITY_FACTORY_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n05 = _city_usd("urban_facade_n05", CITY_PH_URBAN_FACADE_USD_PATH, (83.0, 2.60, CITY_STREET_FACADE_Z), scale=CITY_URBAN_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n06 = _city_usd("factory_facade_n06", CITY_PH_FACTORY_FACADE_USD_PATH, (96.0, 2.20, CITY_STREET_FACADE_Z), scale=CITY_FACTORY_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n07 = _city_usd("urban_facade_n07", CITY_PH_URBAN_FACADE_USD_PATH, (24.0, 2.20, CITY_STREET_FACADE_Z), scale=CITY_URBAN_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n08 = _city_usd("factory_facade_n08", CITY_PH_FACTORY_FACADE_USD_PATH, (37.0, 2.20, CITY_STREET_FACADE_Z), scale=CITY_FACTORY_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)
        city_facade_n09 = _city_usd("urban_facade_n09", CITY_PH_URBAN_FACADE_USD_PATH, (50.0, 2.20, CITY_STREET_FACADE_Z), scale=CITY_URBAN_STREET_FACADE_SCALE, rot=CITY_FACADE_NORTH_ROT)

        city_block_body_s01 = _city_box("city_block_body_s01", (-143.0, -4.20, 1.05), (7.0, 1.00, 2.10), (0.47, 0.42, 0.38), roughness=0.86)
        city_block_body_s02 = _city_box("city_block_body_s02", (-122.0, -4.28, 1.20), (7.8, 1.08, 2.40), (0.34, 0.38, 0.42), roughness=0.82)
        city_block_body_s03 = _city_box("city_block_body_s03", (100.0, -4.20, 1.00), (6.8, 1.00, 2.00), (0.52, 0.46, 0.40), roughness=0.86)
        city_block_body_s04 = _city_box("city_block_body_s04", (123.0, -4.28, 1.18), (7.6, 1.08, 2.36), (0.36, 0.39, 0.42), roughness=0.82)
        city_block_body_s05 = _city_box("city_block_body_s05", (147.0, -4.20, 1.05), (7.0, 1.00, 2.10), (0.49, 0.44, 0.38), roughness=0.86)
        city_facade_s01 = _city_usd("urban_facade_s01", CITY_PH_URBAN_FACADE_USD_PATH, (-43.0, -3.62, CITY_BACK_FACADE_Z), scale=CITY_URBAN_BACK_FACADE_SCALE, rot=CITY_FACADE_SOUTH_ROT)
        city_facade_s02 = _city_usd("factory_facade_s02", CITY_PH_FACTORY_FACADE_USD_PATH, (-22.0, -3.62, CITY_BACK_FACADE_Z), scale=CITY_FACTORY_BACK_FACADE_SCALE, rot=CITY_FACADE_SOUTH_ROT)
        city_facade_s03 = _city_usd("urban_facade_s03", CITY_PH_URBAN_FACADE_USD_PATH, (0.0, -3.62, CITY_BACK_FACADE_Z), scale=CITY_URBAN_BACK_FACADE_SCALE, rot=CITY_FACADE_SOUTH_ROT)
        city_facade_s04 = _city_usd("factory_facade_s04", CITY_PH_FACTORY_FACADE_USD_PATH, (23.0, -3.62, CITY_BACK_FACADE_Z), scale=CITY_FACTORY_BACK_FACADE_SCALE, rot=CITY_FACADE_SOUTH_ROT)
        city_facade_s05 = _city_usd("urban_facade_s05", CITY_PH_URBAN_FACADE_USD_PATH, (47.0, -3.62, CITY_BACK_FACADE_Z), scale=CITY_URBAN_BACK_FACADE_SCALE, rot=CITY_FACADE_SOUTH_ROT)

        city_real_back_n01 = _city_usd("real_building_back_n01", CITY_REAL_BUILDING_USD_PATH, (-74.0, 5.30, 0.0), scale=CITY_REAL_BUILDING_SCALE, rot=CITY_REAL_NORTH_ROT)
        city_real_back_n02 = _city_usd("real_station_back_n02", CITY_REAL_STATION_USD_PATH, (78.0, 5.40, 0.0), scale=CITY_REAL_STATION_SCALE, rot=CITY_REAL_NORTH_ROT)
        city_real_back_s01 = _city_usd("real_building_back_s01", CITY_REAL_BUILDING_USD_PATH, (88.0, -5.70, 0.0), scale=CITY_REAL_BUILDING_SCALE, rot=CITY_REAL_SOUTH_ROT)

        city_tree_trunk_01 = _city_cylinder("tree_trunk_01", (-45.0, 1.64, 0.55), 0.055, 1.10, (0.25, 0.13, 0.06), roughness=0.80)
        city_tree_trunk_02 = _city_cylinder("tree_trunk_02", (-21.0, 1.64, 0.55), 0.055, 1.10, (0.25, 0.13, 0.06), roughness=0.80)
        city_tree_trunk_03 = _city_cylinder("tree_trunk_03", (5.0, 1.64, 0.55), 0.055, 1.10, (0.25, 0.13, 0.06), roughness=0.80)
        city_tree_trunk_04 = _city_cylinder("tree_trunk_04", (31.0, 1.64, 0.55), 0.055, 1.10, (0.25, 0.13, 0.06), roughness=0.80)
        city_tree_trunk_05 = _city_cylinder("tree_trunk_05", (57.0, 1.64, 0.55), 0.055, 1.10, (0.25, 0.13, 0.06), roughness=0.80)
        city_tree_crown_01 = _city_box("tree_crown_01", (-45.0, 1.64, 1.35), (0.90, 0.80, 0.70), (0.12, 0.34, 0.18), roughness=0.70)
        city_tree_crown_02 = _city_box("tree_crown_02", (-21.0, 1.64, 1.35), (0.82, 0.74, 0.66), (0.10, 0.30, 0.16), roughness=0.70)
        city_tree_crown_03 = _city_box("tree_crown_03", (5.0, 1.64, 1.35), (0.86, 0.78, 0.68), (0.13, 0.36, 0.18), roughness=0.70)
        city_tree_crown_04 = _city_box("tree_crown_04", (31.0, 1.64, 1.35), (0.88, 0.80, 0.70), (0.11, 0.32, 0.17), roughness=0.70)
        city_tree_crown_05 = _city_box("tree_crown_05", (57.0, 1.64, 1.35), (0.82, 0.74, 0.66), (0.12, 0.35, 0.18), roughness=0.70)

        city_light_pole_01 = _city_cylinder("streetlight_near_pole_01", (-54.0, 1.72, 1.30), 0.038, 2.60, (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_pole_02 = _city_cylinder("streetlight_near_pole_02", (-30.0, 1.72, 1.30), 0.038, 2.60, (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_pole_03 = _city_cylinder("streetlight_near_pole_03", (-6.0, 1.72, 1.30), 0.038, 2.60, (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_pole_04 = _city_cylinder("streetlight_near_pole_04", (18.0, 1.72, 1.30), 0.038, 2.60, (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_pole_05 = _city_cylinder("streetlight_near_pole_05", (42.0, 1.72, 1.30), 0.038, 2.60, (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_arm_01 = _city_box("streetlight_near_arm_01", (-54.0, 1.32, 2.58), (0.07, 0.80, 0.055), (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_arm_02 = _city_box("streetlight_near_arm_02", (-30.0, 1.32, 2.58), (0.07, 0.80, 0.055), (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_arm_03 = _city_box("streetlight_near_arm_03", (-6.0, 1.32, 2.58), (0.07, 0.80, 0.055), (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_arm_04 = _city_box("streetlight_near_arm_04", (18.0, 1.32, 2.58), (0.07, 0.80, 0.055), (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_arm_05 = _city_box("streetlight_near_arm_05", (42.0, 1.32, 2.58), (0.07, 0.80, 0.055), (0.05, 0.06, 0.07), metallic=0.65, roughness=0.30)
        city_light_head_01 = _city_box("streetlight_near_head_01", (-54.0, 0.94, 2.56), (0.28, 0.18, 0.09), (1.0, 0.78, 0.32), emissive=(1.0, 0.55, 0.16), roughness=0.18)
        city_light_head_02 = _city_box("streetlight_near_head_02", (-30.0, 0.94, 2.56), (0.28, 0.18, 0.09), (1.0, 0.78, 0.32), emissive=(1.0, 0.55, 0.16), roughness=0.18)
        city_light_head_03 = _city_box("streetlight_near_head_03", (-6.0, 0.94, 2.56), (0.28, 0.18, 0.09), (1.0, 0.78, 0.32), emissive=(1.0, 0.55, 0.16), roughness=0.18)
        city_light_head_04 = _city_box("streetlight_near_head_04", (18.0, 0.94, 2.56), (0.28, 0.18, 0.09), (1.0, 0.78, 0.32), emissive=(1.0, 0.55, 0.16), roughness=0.18)
        city_light_head_05 = _city_box("streetlight_near_head_05", (42.0, 0.94, 2.56), (0.28, 0.18, 0.09), (1.0, 0.78, 0.32), emissive=(1.0, 0.55, 0.16), roughness=0.18)

        light = AssetBaseCfg(prim_path="/World/light", spawn=DomeLightCfg(intensity=4200.0, color=(0.82, 0.88, 1.0)))

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=25.0)
    observations: PlatoonObservationsCfg = PlatoonObservationsCfg()
    rewards: PlatoonRewardsCfg = PlatoonRewardsCfg()
    algorithm: PlatoonAlgorithmCfg = PlatoonAlgorithmCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(-5.2, -2.95, 1.38),
        lookat=(-11.75, 0.28, 0.54),
    )

    @configclass
    class ActionCfg:
        joint_vel_1 = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=WHEEL_ACTION_SCALE)
        joint_vel_2 = mdp.JointVelocityActionCfg(asset_name="robot_2", joint_names=[".*_wheel_joint"], scale=WHEEL_ACTION_SCALE)
        joint_vel_3 = mdp.JointVelocityActionCfg(asset_name="robot_3", joint_names=[".*_wheel_joint"], scale=WHEEL_ACTION_SCALE)
        joint_vel_4 = mdp.JointVelocityActionCfg(asset_name="robot_4", joint_names=[".*_wheel_joint"], scale=WHEEL_ACTION_SCALE)
        joint_vel_5 = mdp.JointVelocityActionCfg(asset_name="robot_5", joint_names=[".*_wheel_joint"], scale=WHEEL_ACTION_SCALE)

    actions: ActionCfg = ActionCfg()

    @configclass
    class CommandCfg:
        base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=TARGET_SPEED_RANGE,
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
    # 论文对齐：攻击器/Teacher/安全投影超参数入口。
    # CA-GAN 与 meta-gradient 更新由 task-local MGRS/HAPPO pipeline 使用这些参数。
    # -------------------------------------------------------------------------
    @configclass
    class AttackCfg:
        enable_attack: bool = False
        max_fdi_pos: float = 1.0      # epsilon_p
        max_fdi_acc: float = 0.10     # epsilon_a
        max_dos_rate: float = 0.02    # rho_max

    @configclass
    class TeacherCfg:
        enable_teacher: bool = True
        teacher_lr: float = 1.0e-4
        realism_lambda: float = 0.1

    @configclass
    class SafetyShieldCfg:
        enable_shield: bool = True
        d_crit: float = 0.50
        d_drop: float = 1.45
        # HAPPO actions are still in the pre-URDF-adapter convention here:
        # negative same-sign wheel commands drive this URDF forward.
        brake_action: float = 0.0
        catchup_action: float = -0.35
        catchup_lateral_limit: float = -1.0
        catchup_centerline_limit: float = -1.0
        lateral_tol: float = 0.035
        lateral_crit: float = 0.55
        lateral_turn_gain: float = 0.30
        lateral_turn_clip: float = 0.08
        lateral_velocity_gain: float = 0.10
        centerline_turn_gain: float = 0.25
        centerline_turn_clip: float = 0.08
        centerline_turn_sign: float = 1.0
        first_follower_lateral_gain_scale: float = 1.0
        first_follower_lateral_clip_scale: float = 1.0
        first_follower_lateral_clip_max: float = 0.04
        first_follower_centerline_gain: float = 0.0
        first_follower_centerline_clip: float = 0.0
        pair2_lateral_gain_scale: float = 1.0
        pair2_lateral_clip_scale: float = 1.0
        pair2_lateral_clip_max: float = 0.10
        pair3_lateral_gain_scale: float = 1.0
        pair3_lateral_clip_scale: float = 1.0
        pair3_lateral_clip_max: float = 0.10
        pair4_lateral_gain_scale: float = 1.0
        pair4_lateral_clip_scale: float = 1.0
        pair4_lateral_clip_max: float = 0.10
        forward_bias_gain: float = 0.0
        forward_bias_clip: float = 0.0
        forward_bias_speed_margin: float = 0.02
        forward_bias_min_command: float = 0.0
        forward_bias_min_gap: float = 0.75
        forward_bias_leader_gain_scale: float = 1.0
        forward_bias_leader_clip_scale: float = 1.0

    attack: AttackCfg = AttackCfg()
    teacher: TeacherCfg = TeacherCfg()
    safety_shield: SafetyShieldCfg = SafetyShieldCfg()

    sim: SimulationCfg = SimulationCfg(dt=0.005, use_fabric=True, render_interval=4)
    decimation: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
