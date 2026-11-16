Meta-AMS：单车 clean 实车接入（2026-11-16，ROS1 / 转角rad / 暂定轴距0.30m）

用途与当前边界
==============
这是使用既有 Meta-AMS s0 最终模型的在线闭环接口，第一阶段只部署领车 actor。
输入为实时动捕，输出为独立 guard 监督后的速度与转角/角速度。不是 CSV 指令回放。
已做 CPU 源码一致性、故障注入单测和独立本机 ROS1 回环测试；没有连接实车。
ROS2 适配器已写入，尚未在 ROS2 环境联调。TX2 性能和实车效果尚未验证。
本目录不会启动训练、更新网络或加载训练时的 Teacher/AMS 双 Q。
配置 attack 必须为 off；攻击注入留待 clean 接入验证后单独实现。
用户本轮指定 ROS1、转角按rad、轴距先用0.30m。当前命令默认 --ros 1，配置已选 speed_steering_angle。
“rad吧”和“先按30cm”作为接入假设记录，protocol_confirmed/geometry_confirmed仍等待现场核验。
本次更新的验证记录在 validation/ros1_rad_030m_r2/：26项核心测试通过，隔离ROS1通信确认Angular.Z=delta。
validation/ 下原有报告保留为初版历史证据，其中0.25m/yaw_rate测试参数不是本次接入配置。

一、旧 MATLAB 通信与现在的动作含义
================================
来源：
https://raw.githubusercontent.com/hs997/densact/dddbe56bdcbb55688deec7657a3d234a8490440e/AMS/Proposed_methodROS0724.m
本包 reference/ 下保留原文件及 SHA256。
旧程序订阅 /mocap_position_05，消息 geometry_msgs/PoseStamped。
Position.X、Position.Y 是位置，Position.Z 被当作 yaw 使用；不能按标准 Pose 的高度理解。
新接口默认兼容 xy_yaw_position_z；若改成标准四元数，明确设置 encoding=quaternion。
旧程序发布 /wheeltec_05/cmd_vel：Linear.X=vc，Angular.Z=wc。
旧控制律中的 2.2、0.7 增益不用于本网络；d=0.25 的注释为非完整约束模型转换参数，
不能据此认定实车轴距就是 0.25m。旧 limiter 限值也不是本次实车标定结果。

当前 actor 使用 11 维观测，确定性取高斯均值，输出 [a_v, a_delta]。
原 clean 领车后处理：clip ±0.4 → 原 centerline shield → 再 clip ±0.4。
反归一化为 v=1.2*a_v [m/s]，delta=0.32*a_delta [rad]，相应上界为 0.48m/s、0.128rad。
这不是网络直接输出 [m/s, rad/s]。内部 candidate 的 Angular.Z 永远表示 delta，
该私有 TwistStamped 只用于 policy→guard，不能直接转发给通用 Twist 驱动。

物理底盘输出由 hardware.command_mode 明确选择：
  speed_steering_angle：Linear.X=v；Angular.Z=delta，要求底盘这个自定义话题确实接受转角。
  speed_yaw_rate：Linear.X=v；Angular.Z=v*tan(delta)/L，L 为实测轴距。
steering_sign 用于已标定的转向正负号。delta 为等效前轮转角，不能当作舵机 PWM 或未经标定的舵机轴角。
前轮左右 Ackermann 几何和舵机标定由现有底层实现；本程序不发后轮转角指令。
当前默认 speed_steering_angle、L=0.30m：底盘 Angular.Z 直接发送经过监督后的delta [rad]，不转换为yaw_rate。
L在该模式下用于估算转弯角速度并限幅；不改变网络权重，也不自动补偿与仿真轴距的差异。
现场确认项仍为false，因此配置原样只能做shadow，不能启动实车输出。

二、结构及文件
==============
实时动捕 → 坐标/朝向校准 → 速度估计 → 11D 观测 → 固定 actor → 原领车后处理
                                                               ↓ 私有候选话题
动捕真值 ─────────────────────────────→ 独立 guard → STM32 底盘
键盘 S / stop 服务 ───────────────────→ 锁定停车

core.py：坐标转换、观测、NumPy actor、原领车后处理、guard 状态机。
run_node.py policy：只发布候选动作，永远不创建底盘命令发布器。
run_node.py guard：默认 shadow；仅 --execute 才创建底盘发布器，仍需显式 arm。
ros_io.py：ROS1/ROS2 消息和服务适配。
stop_keyboard.py：S 或空格发软件停车请求；Q 或退出时也请求停车。
benchmark.py：无 ROS、无发布的本机推理计时，适合先在 TX2 执行。
models/meta_ams_s0_leader：仅领车固定权重、尺度/路径配置、来源哈希及导出验证。
tests/test_core.py：离线故障注入测试，不连接 ROS。
tests/validate_against_source.py：训练服务器上的实际源码函数对照工具，需要 PyTorch、PyYAML 和原始源文件。
tests/ros1_loopback.py：仅测试使用的隔离 localhost ROS1 master，不接实际底盘。
validation/：本轮验证报告和离线候选输出；不是实车实验结果。

三、先确认实车信息
==================
1. 已选ROS1，具体发行版及 TX2 的 Python/NumPy 环境待现场核对。运行代码为 Python 3.6+ 语法，不能直接用 Python2 执行。
   运行依赖 numpy，以及对应 Python 的 rospy 或 rclpy、geometry_msgs、std_msgs、std_srvs。
   ROS1/ROS2 的 Python 绑定必须与所用解释器匹配；本包不安装或升级 JetPack/系统 ROS。
   服务器已验证 Python3.13/NumPy2.3 和 Python3.7/NumPy1.21 的核心；未实测 Python3.6/TX2。
2. 确认当前 cmd_vel 的 Angular.Z 到 STM32 的实际语义、单位和转向正负号。
3. 测量轴距 L、车体长宽；确认实际等效转角范围、速度死区和制动性能。
4. 确认动捕 yaw 单位、frame_id、频率和 header.stamp 是否为真实采集时间。
   多机时钟必须同步。不能把冻结的旧位姿反复盖上当前时间，否则会掩盖数据陈旧。
5. 实测路径朝向、起点、可用场地边界和动捕标记相对模型车体参考点的偏移。
6. 实测 STM32/底盘在停止收到命令时能自主停车；保留可触及的物理停止手段。
   software stop、Python finally 都不能处理 TX2 断电、guard 被 SIGKILL 或链路全断。
   base_command_timeout_verified 只能在真实测试通过后确认，本包没有修改 STM32 固件。

上述校准填入 config/single_car_clean.json 的副本。不要仅把 confirmed 改为 true 来绕过检查。
protocol_confirmed、geometry_confirmed、transform_confirmed、geofence_confirmed 都是实测确认状态。
vehicle_length/width、geofence 当前为模板值，不是本实验室场地或实车测量结果。
特别是旧模板车长0.24m小于暂定轴距0.30m，不能用于这辆实车；需填写实测车体包络，不能由轴距猜车长。
guard会拒绝车长小于轴距的配置，即使误将geometry_confirmed设为true也不能arm。
速度初始限值 0.15m/s、加速度 0.20m/s²、转角速度 0.20rad/s、角速度 0.35rad/s 是待标定调试设置，
不构成已验证安全阈值。这些新增限制会改变闭环行为，不能把实车表现与仿真表直接视为同协议。

四、观测与场地对齐
==================
领车观测：[0,0,0,0, vx_body,vy_body,yaw_rate, clip(y-y_ref,-2,2), sin(yaw-heading_ref),0,0]。
模型的领车没有前车，使用 actor[0]、role=0。不能把跟随车 role 或前车输入随意置零后冒充五车验证。
动捕差分得到世界系速度，经滤波和旋转后转为车体系；yaw 差分先处理 ±pi 跨越。
滤波/延迟/噪声与仿真的真实状态仍有差别，必须通过实测反馈判断是否需要硬件匹配的仿真微调。

参考路径保留已训练的空间五次曲线：x=-7.75 开始，长度6m，y 从 -0.065 到 -1.160。
默认领车起点 (-8.75,-0.065)，起点到变道段末端约7m，之后还需稳定及停车余量。
origin_world_xy 是实车参考点的实测起点，映射到 origin_model_xy=(-8.75,-0.065)。
path_heading_world_rad 是模型 +x 在动捕世界系中的朝向。程序只做平移和旋转，不缩放空间。
body_to_marker_xy_m 是车体参考点指向动捕标记的向量；marker_to_body_yaw_rad 是朝向偏置。
两者用于把标记位姿还原到与训练状态对应的参考点。不能不加区分地用任意标记中心替代。
原仿真轴距0.14325m只留作来源信息，不能拿来代替实车L。
若实际轴距、速度响应或场地与仿真明显不同，先匹配仿真模型并复验/微调，再做实车策略验收。
单纯换算 omega=v*tan(delta)/L 只解决接口单位，不能自动消除动力学差异。

五、TX2 首次操作顺序
===================
复制本目录，在正确 ROS/Python 环境下进入目录。下面路径都是相对本目录。
先执行不连接底盘的测试和计时：
  python3 tests/test_core.py
  python3 benchmark.py --model models/meta_ams_s0_leader
计时未含 ROS 传输；检查实际端到端延迟是否满足50Hz和配置的超时门限。

第一次只启动 policy，看实时坐标、观测和候选输出，不向底盘发布命令：
  python3 run_node.py policy --ros 1 --config config/single_car_clean.json --model models/meta_ams_s0_leader --log-dir runs/policy_shadow_001
另一个终端可启动 shadow guard：
  python3 run_node.py guard --ros 1 --config config/single_car_clean.json --model models/meta_ams_s0_leader --log-dir runs/guard_shadow_001
ROS2 把 --ros 1 改成 --ros 2；当前 ROS2 适配器尚待实际系统联调。
每次 --log-dir 必须是新目录。重用目录会直接退出，避免覆盖旧数据。
policy 和 guard 要使用相同配置/模型；候选消息带有配置+模型指纹，混用会停车。

先完成校准、零指令检查、轮子悬空的方向检查，以及独立底盘超时停车测试。
实车低速试验前停掉 MATLAB、move_base、键盘遥控等其他底盘发布者，并停止 shadow guard。
同一主机/底盘 topic 的重复 guard 会被文件锁拒绝；其他主机上的发布者仍需人工核对。
在 policy 持续输出且有足够现场空间时，用实测配置启动真实 guard（此命令只在车旁执行）：
  python3 run_node.py guard --ros 1 --config config/single_car_clean.json --model models/meta_ams_s0_leader --log-dir runs/guard_clean_001 --execute
另一个本地终端提前启动停车键盘：
  python3 stop_keyboard.py --ros 1 --config config/single_car_clean.json
guard 初始 DISARMED，持续发布零命令；不自动起步。检查状态后显式启动：
  rosservice call /ams/car05/arm "{}"
软件停车：按 S/空格，或
  rosservice call /ams/car05/stop "{}"
ROS2 对应服务命令：
  ros2 service call /ams/car05/arm std_srvs/srv/Trigger "{}"
  ros2 service call /ams/car05/stop std_srvs/srv/Trigger "{}"
上述服务名固定 car05，当前只接一辆车；不能直接多开并冒充完整五车部署。

guard 只在起点附近、朝向正确、已静止、位姿新鲜且所有实测条件确认后接受 arm。
位置时间戳缺失/过旧、策略断流、guard 调度延迟、异常值、车体越界、过大跟踪误差等会锁定停车。
恢复数据不会自行恢复行驶，需要回到合格起始状态并显式重新 arm。
原路径完成的单车条件是 progress>=0.99、横向误差<=0.12m、航向误差<=0.12rad，保持2s后停车。
这只是单车条件，不含五车完成/间距条件，不能记作五车实验通过。
Ctrl+C/SIGTERM 正常退出时连续发零命令；底盘自身的断令停车负责进程被强杀和主机故障情形。

六、记录与之后的攻击实验
======================
每次新建 session.json、samples.csv、result.json，失败/提前停车的数据也保留。
policy 记录观测、网络原始动作、原 shield 后动作；guard 记录实际发出的限幅后速度/转角/第二通道、
限制介入标记、停车原因与单车完成状态。命令不是实测舵角/车速；如底盘有反馈应另录 ROS bag。
实车拍摄真实变道视频并同步保存动捕/底盘反馈、guard 状态和停车话题，不能用旧仿真视频替代。
首次 clean 验收先检查完整路径、误差、真实车体边界、停止距离和监督介入率，再决定扩展。

下一步顺序：
  A. 单车 clean：当前接口完成硬件校准、shadow、悬空方向验证、低速闭环。
  B. 两车 clean：部署真实跟随车 actor、前车观测及对应后处理；验证共同坐标和数据时间对齐。
  C. 五车 clean：恢复五个 actor/角色及完整观测/安全盾，检验全车依次通过同一空间路径和车距。
  D. 应用层受控攻击：在提供给策略的观测/候选指令支路上模拟有界FDI、丢包、延迟和混合扰动，
     同步保存原始、受扰、策略、guard 后动作及攻击配置/种子。独立动捕真值与停车链路保持可用。
     注入位置、掩码、保持/置零策略和强度要逐一对齐当前训练 attacker，不能只叫同一个名字。
     从低强度开始，每档同时做 clean 回归；不得把无线干扰或阻断急停当作攻击实现方式。
  E. 相同实车协议比较 Meta-AMS、消融和传统控制器，报告所有失败及监督介入，区分训练种子和重复实车试验。

当前单车 clean 不能证明五车协同或混合攻击鲁棒性；也未开始实车在线探索训练。
本包不会因命令行参数启用攻击。本次没有改论文、旧模型、原实验配置或结果。
