%% 加密控制方法：起点位置/朝向归位 + 二次确认 + 正式实验数据保存
% 功能说明：
% 1. 正式实验前，小车先移动到期望起点后方的“预对准点”。
% 2. 再从预对准点沿期望轨迹初始方向驶向 xd(1), yd(1)，同时校正初始航向。
% 3. 位置和航向均满足要求后，小车停车并等待二次确认。
% 4. 按 Space（空格）或 Enter（回车）后才开始正式加密控制。
% 5. 归位和等待确认数据不保存；二次确认后也只有完整跑完全部控制步才保存。
% 6. 全程均可按 S 键紧急停车。中途按 S 只停车退出，不保存任何实验数据。
% 7. 正式控制不人为补齐 Ts，按 MATLAB、动捕和 ROS 的实际节奏运行。
% 8. 期望轨迹在 90 s 内严格运行一圈八字：t=0 从 (0,2) 出发，t=T 回到 (0,2) 后停车。
% 9. 数据目录按“方法名+延迟”建立，例如：Proposed_method+40delay。
%
% 重要按键：
%   S            紧急停车并退出
%   Space/Enter  归位完成后确认正式开始

close all;
clearvars;
clc;

%% ==================== 全局状态变量 ====================
global stopSimulation cmdpub runConfirmed waitingForConfirmation
stopSimulation = false;
runConfirmed = false;
waitingForConfirmation = false;

%% ==================== 实验名称及数据保存配置 ====================
% 修改方法时，只需要改 method_name。
% 例如：'Proposed_method'、'Compared_method'。
method_name = 'Proposed_method';

% 文件夹名称默认按 tau 的“延迟步数”命名。
% 例如 tau=0  -> Proposed_method+0delay
%      tau=5  -> Proposed_method+5delay
%      tau=40 -> Proposed_method+40delay
save_root = fullfile(pwd, 'experiment_data');

%% ==================== 仿真系统和参数设置 ====================
T = 90;                          % 一圈八字轨迹的名义总时长，s
Ts = 0.1;                        % 名义采样周期，s
n = round(T / Ts) + 1;           % 含 t=0 和 t=T，共 901 个采样点
num_control_intervals = n - 1;   % 实际下发非零控制的区间数，共 900 个
tau = 30;                        % 延迟步数

% 实际延迟同时保存在元数据中
% delay_seconds = tau * Ts

d = 0.25;                % 非完整约束模型转换参数
v1m = 0.1;                % 原参数，保留
v2m = 0.1;                % 原参数，保留

% 生成实验标签，例如 Proposed_method+40delay
safe_method_name = regexprep(method_name, '[^a-zA-Z0-9_-]', '_');
experiment_tag = sprintf('%s+%ddelay', safe_method_name, tau);
save_directory = fullfile(save_root, experiment_tag);

%% ==================== 给定控制增益 K，求解 P ====================
k1 = 0.9;
k2 = 0.7;
K = [k1, 0; 0, k2];

A = eye(2);
B = Ts * eye(2);
Phi = (A - B*K)^tau;

A1 = zeros(2*(tau+1), 2*(tau+1));
A1(1:2, 1:2) = A;
A1(1:2, end-1:end) = A1(1:2, end-1:end) - B*K*Phi;

for i = 1:tau
    row_start = i*2 + 1;
    col_start = (i-1)*2 + 1;
    A1(row_start:row_start+1, col_start:col_start+1) = eye(2);
end

eig_A1 = eig(A1);
if any(abs(eig_A1) >= 1)
    error('A1 is not Schur stable.');
end

Q = eye(2*(tau+1));
P = dlyap(A1', Q);

ifright = (eig(P) > 0).';
disp('P 的正定性检查结果：');
disp(ifright);

B1 = zeros(2*(tau+1), 2);
B1(end-1:end, :) = B*K;

B2 = zeros(2*(tau+1), 2);
B2(end-1:end, :) = B;

B3 = zeros(2, 2*(tau+1));
B3(:, end-1:end) = Phi;

%% Young 不等式参数
epsilon1 = 0.01;
epsilon2 = 0.01;
epsilon3 = 0.01;
epsilon4 = 0.01;

%% 收敛率参数
alpha1 = 0.1;
alpha2 = 0.2;
alpha3 = 0.3; %#ok<NASGU>
alpha4 = 0.4;

f1 = norm(B2'*P*B2*B3*B3') ...
    + 2*norm(A1'*P*B2*B3) ...
    + epsilon2*norm(B1'*P*B2*B3*B3') ...
    + epsilon3*norm(A1'*P*B2);

f2 = norm(B1'*P*B1) ...
    + norm(A1'*P*B1)/epsilon1 ...
    + norm(B1'*P*B2)/epsilon2 ...
    + norm(A1'*P*B2)/epsilon3;

f3 = f2 ...
    + norm(B2'*P*B2) ...
    + alpha1*norm(B2'*P*B2*B3)/epsilon4 ...
    + 2*sqrt(alpha1)*norm(B1'*P*B2);

f4 = epsilon1*norm(A1'*P*B1) ...
    + epsilon4*norm(B2'*P*B2*B3);

disp('[f1, f2, f3, f4] =');
disp([f1, f2, f3, f4]);

if f4 < alpha4
    disp('parameter correct');
else
    warning('当前参数未满足 f4 < alpha4。');
end

%% ==================== 量化 K ====================
D1 = sqrt(alpha1 * min(eig(Q)) / f1);
DeltaK = D1 - floor(D1*1000)/1000;

if DeltaK <= 0
    error('DeltaK <= 0，无法进行量化，请检查参数。');
end

Ksat = 10000;
QK = K_Uniform_Quantizer_Encode(K, Ksat, DeltaK);

%% ==================== Paillier 密钥 ====================
bit_length = 64;         % 可设为 64、128、256、512 等
p = PaillierCrypto(bit_length);
keys = p.generateKeys(); %#ok<NASGU>
n2 = p.getPublicKey.n2;

%% ==================== 严格一圈八字轨迹 ====================
% 时间向量严格包含起点 t=0 和终点 t=T。
% 轨迹在 t=0 与 t=T 时均为 (0, 2)，因此正好完成一圈八字。
t_ref = (0:n-1).' * Ts;

xd = sin(4*pi*t_ref/T);
yd = 2*cos(2*pi*t_ref/T);

% 使用解析导数生成期望转换模型速度，避免终点差分为零或数值偏差。
taud = zeros(n,2);
taud(:,1) = (4*pi/T) * cos(4*pi*t_ref/T);
taud(:,2) = -(4*pi/T) * sin(2*pi*t_ref/T);

% 期望轨迹切向航向。起点和终点航向均为 0 rad。
betad = unwrap(atan2(taud(:,2), taud(:,1)));

fprintf('八字轨迹检查：起点 (%.6f, %.6f)，终点 (%.6f, %.6f)\n', ...
    xd(1), yd(1), xd(end), yd(end));
fprintf('控制区间数：%d，轨迹采样点数：%d\n', ...
    num_control_intervals, n);

%% ==================== ROS 发布、订阅及按键监听 ====================
cmdpub = rospublisher('/wheeltec_05/cmd_vel', rostype.geometry_msgs_Twist);
stopRobot(cmdpub);

% 脚本正常结束、报错或中断时，都会再次发送停车命令
cleanupObject = onCleanup(@() stopRobot(cmdpub)); %#ok<NASGU>

sub_name = rossubscriber( ...
    '/mocap_position_05', ...
    'geometry_msgs/PoseStamped', ...
    'DataFormat', 'struct');

controlFigure = figure( ...
    'Name', 'Robot Control: S=STOP, Space/Enter=START', ...
    'NumberTitle', 'off');
set(controlFigure, ...
    'WindowKeyPressFcn', @(src,event) onKeyPressed(event), ...
    'CloseRequestFcn', @(src,event) onControlFigureClosed(src));

fprintf('\nROS 发布器正在初始化，请稍候……\n');
pauseWithStop(3);

if stopSimulation
    stopRobot(cmdpub);
    fprintf('初始化阶段收到 S 键停车指令，程序退出。\n');
    return;
end

%% ==================== 起点位置及朝向归位 ====================
% 预对准点位于期望起点沿初始航向反方向 alignment_distance 处。
% 小车先到预对准点，再向期望起点前进，可显著改善到点时的初始朝向。
alignment_distance = 0.75;             % 预对准距离，m，增大直线对准距离
alignment_position_tolerance = 0.15;   % 预对准点容差，m，避免低速死区附近往复振荡
start_position_tolerance = 0.05;       % 期望起点位置容差，m
start_yaw_tolerance = 8*pi/180;        % 期望起点航向容差，rad
start_max_time = 150;                  % 单次完整归位最大时间，s

% 如果等待确认期间车辆发生明显漂移，则自动重新归位，再次等待确认。
confirmation_position_tolerance = 0.08;
confirmation_yaw_tolerance = 12*pi/180;

while true
    fprintf('\n========================================\n');
    fprintf('开始执行起点位置及朝向归位\n');
    fprintf('期望起点：x = %.4f m, y = %.4f m\n', xd(1), yd(1));
    fprintf('期望初始航向：%.2f deg\n', betad(1)*180/pi);
    fprintf('预对准距离：%.2f m\n', alignment_distance);
    fprintf('全程可按 S 键立即停车\n');
    fprintf('========================================\n');

    reached_start = moveToDesiredStartAligned( ...
        cmdpub, sub_name, ...
        xd(1), yd(1), betad(1), ...
        alignment_distance, ...
        alignment_position_tolerance, ...
        start_position_tolerance, ...
        start_yaw_tolerance, ...
        start_max_time);

    if stopSimulation
        stopRobot(cmdpub);
        fprintf('归位阶段收到 S 键停车指令，程序退出。\n');
        return;
    end

    if ~reached_start
        stopRobot(cmdpub);
        error('规定时间内未完成起点位置及朝向归位，正式控制未启动。');
    end

    %% 二次确认：此时始终保持零速度
    stopRobot(cmdpub);
    runConfirmed = false;
    waitingForConfirmation = true;

    fprintf('\n========================================\n');
    fprintf('小车已经到达期望起点，并完成初始朝向对准。\n');
    fprintf('请确认场地安全及小车状态。\n');
    fprintf('按 Space（空格）或 Enter（回车）开始正式实验。\n');
    fprintf('按 S 键立即停车并退出。\n');
    fprintf('注意：确认前的数据不会保存。\n');
    fprintf('========================================\n');

    waitForRunConfirmation();
    waitingForConfirmation = false;

    if stopSimulation
        stopRobot(cmdpub);
        fprintf('等待二次确认时收到 S 键，程序退出。\n');
        return;
    end

    %% 确认后重新读取动捕，并检查等待期间是否发生漂移
    [msg0, mocap_ok] = receiveMocapWithStop(sub_name, 10);
    if ~mocap_ok
        stopRobot(cmdpub);
        if stopSimulation
            fprintf('读取正式实验初始状态时收到 S 键，程序退出。\n');
            return;
        else
            error('二次确认后 10 s 内未收到动捕数据。');
        end
    end

    x0 = msg0.Pose.Position.X;
    y0 = msg0.Pose.Position.Y;
    beta0 = msg0.Pose.Position.Z;

    confirmation_position_error = hypot(x0-xd(1), y0-yd(1));
    confirmation_yaw_error = abs(wrapToPi(beta0-betad(1)));

    if confirmation_position_error <= confirmation_position_tolerance && ...
            confirmation_yaw_error <= confirmation_yaw_tolerance
        fprintf('\n二次确认有效，正式实验即将开始。\n');
        fprintf('正式初始位置误差：%.4f m\n', confirmation_position_error);
        fprintf('正式初始航向误差：%.2f deg\n', ...
            confirmation_yaw_error*180/pi);
        break;
    end

    fprintf('\n等待确认期间小车位置或朝向发生偏移：\n');
    fprintf('位置误差：%.4f m\n', confirmation_position_error);
    fprintf('航向误差：%.2f deg\n', confirmation_yaw_error*180/pi);
    fprintf('程序将自动重新归位，归位后需要再次二次确认。\n');
    stopRobot(cmdpub);
    pauseWithStop(1);
end

if stopSimulation
    stopRobot(cmdpub);
    return;
end

%% ==================== 正式实验变量定义 ====================
% 下面所有数组都只对应“二次确认以后”的正式实验数据。
xp = zeros(n,1);
yp = zeros(n,1);
betap = zeros(n,1);

V1p = zeros(n,1);
V2p = zeros(n,1);
v1p = zeros(n,1);
v2p = zeros(n,1);
vcp = zeros(n,1);
wcp = zeros(n,1);

Xsat = 100000;
Deltax = zeros(n,1);
Qxphat = zeros(n,1);
Qyphat = zeros(n,1);
Qxpdhat = zeros(n,1);
Qypdhat = zeros(n,1);
Realv1p = zeros(n,1);
Realv2p = zeros(n,1);

xe = zeros(n,1);
ye = zeros(n,1);
position_error = zeros(n,1);
theta_des = zeros(n,1);
theta_error = zeros(n,1);

% 真实执行时间和计算耗时数据
time_actual = zeros(n,1);
mocap_receive_time = zeros(n,1);
crypto_time = zeros(n,1);
loop_time = zeros(n,1);

% 使用二次确认后的第一帧动捕作为正式控制初值
xp(1) = msg0.Pose.Position.X;
yp(1) = msg0.Pose.Position.Y;
betap(1) = msg0.Pose.Position.Z;

%% ==================== 转换模型预测状态 ====================
xehat = zeros(n,1);
yehat = zeros(n,1);
xphat = zeros(n,1);
yphat = zeros(n,1);

xphat(1) = xp(1);
yphat(1) = yp(1);
xehat(1) = xp(1) - xd(1);
yehat(1) = yp(1) - yd(1);

BB = A - B*K;
for k = 1:n-1
    e_next = BB * [xehat(k); yehat(k)];
    xehat(k+1) = e_next(1);
    yehat(k+1) = e_next(2);
    xphat(k+1) = xehat(k+1) + xd(k+1);
    yphat(k+1) = yehat(k+1) + yd(k+1);
end

V1p(1) = -K(1,1)*(xphat(1)-xd(1)) + taud(1,1);
V2p(1) = -K(2,2)*(yphat(1)-yd(1)) + taud(1,2);

%% ==================== 正式加密控制循环 ====================
fprintf('\n========================================\n');
fprintf('正式加密控制开始：%s\n', experiment_tag);
fprintf('从此刻开始记录实验数据。\n');
fprintf('按 S 键可立即停车；中途停止不会保存任何实验数据。\n');
fprintf('========================================\n');

formal_start_timer = tic;
last_k = 0;
formal_error = [];

try
    for k = 1:n
        step_timer = tic;

        % 与原始实验代码保持一致：每步先处理按键事件，再暂停 0.01 s
        drawnow;
        pause(0.01);

        if stopSimulation
            stopRobot(cmdpub);
            fprintf('正式控制阶段收到 S 键，控制循环停止。\n');
            break;
        end

        %% ---------- 1. 读取动捕 ----------
        mocap_timer = tic;
        [msg, mocap_ok] = receiveMocapWithStop(sub_name, 10);
        mocap_receive_time(k) = toc(mocap_timer);

        if ~mocap_ok
            stopRobot(cmdpub);

            if stopSimulation
                fprintf('等待动捕期间收到 S 键，控制循环停止。\n');
                break;
            else
                warning('10 s 内未收到动捕数据，已停车并结束正式控制。');
                break;
            end
        end

        time_actual(k) = toc(formal_start_timer);
        xp(k) = msg.Pose.Position.X;
        yp(k) = msg.Pose.Position.Y;

        %% ---------- 2. 根据延迟状态更新预测状态 ----------
        if tau == 0
            xphat(k) = xp(k);
            yphat(k) = yp(k);
        elseif k - tau >= 1
            e_delay = [xp(k-tau) - xd(k-tau); ...
                       yp(k-tau) - yd(k-tau)];

            e_hat = Phi * e_delay;

            xphat(k) = xd(k) + e_hat(1);
            yphat(k) = yd(k) + e_hat(2);
        else
            xphat(k) = xp(k);
            yphat(k) = yp(k);
        end

        %% ---------- 3. 航向连续化 ----------
        yawRaw = msg.Pose.Position.Z;

        if k == 1
            betap(k) = yawRaw;
        else
            deltaYaw = wrapToPi(yawRaw - betap(k-1));
            betap(k) = betap(k-1) + deltaYaw;
        end

        %% ---------- 4. 误差计算 ----------
        xe(k) = xd(k) - xp(k);
        ye(k) = yd(k) - yp(k);
        position_error(k) = hypot(xe(k), ye(k));

        theta_des(k) = atan2(ye(k), xe(k));
        theta_error(k) = wrapToPi(theta_des(k) - betap(k));

        %% ---------- 5. 状态量化及 Paillier 运算 ----------
        crypto_timer = tic;

        if k - tau >= 1
            X = xp(k:-1:k-tau).';
            Y = yp(k:-1:k-tau).';
            XY = [X, Y];

            D2 = 2*sqrt(alpha2*min(eig(Q))/(f2+f3))*norm(XY);
            Deltax(k) = D2 - floor(D2*1000)/1000;

            if Deltax(k) <= 0
                Deltax(k) = 0.001;
            end

            Qxphat(k) = X_Uniform_Quantizer_Encode( ...
                xphat(k), Xsat, Deltax(k));
            Qyphat(k) = X_Uniform_Quantizer_Encode( ...
                yphat(k), Xsat, Deltax(k));

            % 保留加密计算过程。密文对象不写入 CSV，量化值和解密结果会保存。
            Exhat = Paillier_encryption(Qxphat(k), Xsat, p); %#ok<NASGU>
            Eyhat = Paillier_encryption(Qyphat(k), Xsat, p); %#ok<NASGU>

            Qxpdhat(k) = X_Uniform_Quantizer_Encode( ...
                xphat(k)-xd(k), Xsat, Deltax(k));
            Qypdhat(k) = X_Uniform_Quantizer_Encode( ...
                yphat(k)-yd(k), Xsat, Deltax(k));

            [Ev1p, Realv1p(k)] = Paillier_add_multpy( ...
                QK(1), Qxpdhat(k), Xsat, p, n2); %#ok<NASGU>
            [Ev2p, Realv2p(k)] = Paillier_add_multpy( ...
                QK(4), Qypdhat(k), Xsat, p, n2); %#ok<NASGU>
        end

        crypto_time(k) = toc(crypto_timer);

        %% ---------- 6. 转换模型控制量 ----------
        if k - tau >= 1
            % 负号已在加密运算中处理
            v1p(k) = DeltaK*Deltax(k)*Realv1p(k);
            V1p(k) = v1p(k) + taud(k,1);

            v2p(k) = DeltaK*Deltax(k)*Realv2p(k);
            V2p(k) = v2p(k) + taud(k,2);
        else
            V1p(k) = V1p(1);
            V2p(k) = V2p(1);
        end

        %% ---------- 7. 转换为实际线速度和角速度 ----------
        Rp = [cos(betap(k)), -d*sin(betap(k)); ...
              sin(betap(k)),  d*cos(betap(k))];

        vwcp = Rp \ [V1p(k); V2p(k)];

        vcp(k) = 2.2*vwcp(1);
        wcp(k) = 0.7*vwcp(2);
        [vcp(k), wcp(k)] = limiter(vcp(k), wcp(k));

        %% ---------- 8. 发送正式控制指令 ----------
        drawnow;
        if stopSimulation
            stopRobot(cmdpub);
            fprintf('发送控制命令前收到 S 键，控制循环停止。\n');
            break;
        end

        if k <= num_control_intervals
            % 前 num_control_intervals 个区间正常下发控制。
            sendVelocity(cmdpub, vcp(k), wcp(k));
        else
            % 第 n 个采样点对应 t=T，只采集终点状态并立即停车。
            stopRobot(cmdpub);
        end

        last_k = k;

        fprintf(['k = %4d/%4d | t_{nom} = %7.3f | t_{act} = %7.3f | ', ...
                 'x = % .4f | y = % .4f | v = % .4f | w = % .4f | ', ...
                 'e = %.4f | crypto = %.4f s\n'], ...
            k, n, (k-1)*Ts, time_actual(k), xp(k), yp(k), ...
            vcp(k), wcp(k), position_error(k), crypto_time(k));

        % 不强制补齐 Ts，按 MATLAB、动捕和 ROS 的实际执行节奏运行。

        loop_time(k) = toc(step_timer);
    end
catch ME
    formal_error = ME;
    stopRobot(cmdpub);
    warning('正式控制出现异常，已停车；本次实验数据不会保存：%s', ME.message);
end

%% 正式控制结束后确保停车
stopRobot(cmdpub);

% 只有采集到 t=T 的第 n 个终点状态、未按 S、且未发生异常，才认定为完整一圈实验。
formal_completed = (last_k == n) && ~stopSimulation && isempty(formal_error);

%% ==================== 保存正式实验数据 ====================
saved_files = struct('mat', '', 'csv', '', 'figure', '', 'png', '');

if formal_completed
    valid = (1:last_k).';

    % 核心时序数据，只截取正式实验的有效长度
    data = struct();
    data.k = valid;
    data.time_nominal = (valid-1)*Ts;
    data.time_actual = time_actual(valid);

    data.xd = xd(valid);
    data.yd = yd(valid);
    data.betad = betad(valid);

    data.xp = xp(valid);
    data.yp = yp(valid);
    data.betap = betap(valid);

    data.xe = xe(valid);
    data.ye = ye(valid);
    data.position_error = position_error(valid);
    data.theta_des = theta_des(valid);
    data.theta_error = theta_error(valid);

    data.xphat = xphat(valid);
    data.yphat = yphat(valid);
    data.xehat = xehat(valid);
    data.yehat = yehat(valid);

    data.taud1 = taud(valid,1);
    data.taud2 = taud(valid,2);
    data.V1p = V1p(valid);
    data.V2p = V2p(valid);
    data.v1p = v1p(valid);
    data.v2p = v2p(valid);
    data.vcp = vcp(valid);
    data.wcp = wcp(valid);

    % 量化及同态计算相关数据
    data.Deltax = Deltax(valid);
    data.Qxphat = Qxphat(valid);
    data.Qyphat = Qyphat(valid);
    data.Qxpdhat = Qxpdhat(valid);
    data.Qypdhat = Qypdhat(valid);
    data.Realv1p = Realv1p(valid);
    data.Realv2p = Realv2p(valid);
    data.quantization_active = valid > tau;

    % 时间性能数据
    data.mocap_receive_time = mocap_receive_time(valid);
    data.crypto_time = crypto_time(valid);
    data.loop_time = loop_time(valid);

    metadata = struct();
    metadata.method_name = method_name;
    metadata.experiment_tag = experiment_tag;
    metadata.created_at = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    metadata.data_scope = 'Only data after Space/Enter confirmation';
    metadata.completed_steps = last_k;
    metadata.requested_steps = n;
    metadata.control_intervals = num_control_intervals;
    metadata.trajectory_description = 'Exactly one figure-eight loop in 90 s from t=0 to t=T';
    metadata.nominal_period_enforced = false;
    metadata.stopped_by_S = stopSimulation;
    metadata.completed_normally = formal_completed;
    metadata.formal_error_message = '';
    if ~isempty(formal_error)
        metadata.formal_error_message = formal_error.message;
    end

    parameters = struct();
    parameters.T = T;
    parameters.Ts = Ts;
    parameters.n = n;
    parameters.num_control_intervals = num_control_intervals;
    parameters.tau_steps = tau;
    parameters.delay_seconds = tau*Ts;
    parameters.d = d;
    parameters.K = K;
    parameters.QK = QK;
    parameters.DeltaK = DeltaK;
    parameters.Ksat = Ksat;
    parameters.Xsat = Xsat;
    parameters.bit_length = bit_length;
    parameters.epsilon = [epsilon1, epsilon2, epsilon3, epsilon4];
    parameters.alpha = [alpha1, alpha2, alpha3, alpha4];
    parameters.f = [f1, f2, f3, f4];
    parameters.alignment_distance = alignment_distance;
    parameters.start_position_tolerance = start_position_tolerance;
    parameters.start_yaw_tolerance = start_yaw_tolerance;

    if ~exist(save_directory, 'dir')
        mkdir(save_directory);
    end

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    file_base = sprintf('%s_%s', experiment_tag, timestamp);

    saved_files.mat = fullfile(save_directory, [file_base, '.mat']);
    saved_files.csv = fullfile(save_directory, [file_base, '.csv']);

    save(saved_files.mat, 'data', 'metadata', 'parameters', '-v7.3');

    data_table = table( ...
        data.k, data.time_nominal, data.time_actual, ...
        data.xd, data.yd, data.betad, ...
        data.xp, data.yp, data.betap, ...
        data.xe, data.ye, data.position_error, ...
        data.theta_des, data.theta_error, ...
        data.xphat, data.yphat, data.xehat, data.yehat, ...
        data.taud1, data.taud2, ...
        data.V1p, data.V2p, data.v1p, data.v2p, ...
        data.vcp, data.wcp, ...
        data.Deltax, data.Qxphat, data.Qyphat, ...
        data.Qxpdhat, data.Qypdhat, ...
        data.Realv1p, data.Realv2p, data.quantization_active, ...
        data.mocap_receive_time, data.crypto_time, data.loop_time, ...
        'VariableNames', { ...
        'k', 'time_nominal', 'time_actual', ...
        'xd', 'yd', 'betad', ...
        'xp', 'yp', 'betap', ...
        'xe', 'ye', 'position_error', ...
        'theta_des', 'theta_error', ...
        'xphat', 'yphat', 'xehat', 'yehat', ...
        'taud1', 'taud2', ...
        'V1p', 'V2p', 'v1p', 'v2p', ...
        'vcp', 'wcp', ...
        'Deltax', 'Qxphat', 'Qyphat', ...
        'Qxpdhat', 'Qypdhat', ...
        'Realv1p', 'Realv2p', 'quantization_active', ...
        'mocap_receive_time', 'crypto_time', 'loop_time'});

    writetable(data_table, saved_files.csv);

    fprintf('\n========================================\n');
    fprintf('正式实验已完整运行，数据已保存。归位数据未写入文件。\n');
    fprintf('实验标签：%s\n', experiment_tag);
    fprintf('MAT 文件：%s\n', saved_files.mat);
    fprintf('CSV 文件：%s\n', saved_files.csv);
    fprintf('有效正式数据点：%d\n', last_k);
    fprintf('========================================\n');
else
    fprintf('本次正式实验未采集到第 %d 个终点采样点，不保存任何数据或轨迹图。\n', n);
    if stopSimulation
        fprintf('原因：运行过程中按下了 S 键。\n');
    elseif ~isempty(formal_error)
        fprintf('原因：正式控制发生异常：%s\n', formal_error.message);
    elseif last_k < n
        fprintf('原因：控制循环提前结束，仅完成 %d/%d 步。\n', last_k, n);
    end
end

%% ==================== 绘图并保存 ====================
if formal_completed
    trajectoryFigure = figure(2);

    h1 = plot( ...
        xd(1:last_k), yd(1:last_k), ...
        'Color', [0.80,0.80,0.80], ...
        'LineStyle', '-', ...
        'LineWidth', 6);
    hold on;

    h2 = plot( ...
        xp(1:last_k), yp(1:last_k), ...
        'Color', [1,0,0], ...
        'LineStyle', '--', ...
        'LineWidth', 3);

    h3 = plot( ...
        xp(1), yp(1), ...
        'p', ...
        'MarkerSize', 15, ...
        'Color', [0,0,1], ...
        'MarkerFaceColor', [0,0,1]);

    set(gca, 'XMinorTick', 'on', 'YMinorTick', 'on', 'FontSize', 20);
    ylabel('$Y$ (m)', ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 26);
    xlabel('$X$ (m)', ...
        'Interpreter', 'latex', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 26);

    grid minor;
    axis([-1.5, 1.5, -3, 4]);

    method_legend = strrep(method_name, '_', ' ');
    h = legend( ...
        [h1,h2,h3], ...
        {'Desired trajectory', method_legend, 'Formal initial position'});
    set(h, ...
        'FontSize', 20, ...
        'FontWeight', 'normal', ...
        'Interpreter', 'none');

    title(sprintf('%s, delay = %d steps (%.2f s)', ...
        method_legend, tau, tau*Ts), ...
        'Interpreter', 'none');

    hold off;

    if ~exist(save_directory, 'dir')
        mkdir(save_directory);
    end

    if ~exist('file_base', 'var')
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        file_base = sprintf('%s_%s', experiment_tag, timestamp);
    end

    saved_files.figure = fullfile(save_directory, [file_base, '_trajectory.fig']);
    saved_files.png = fullfile(save_directory, [file_base, '_trajectory.png']);

    savefig(trajectoryFigure, saved_files.figure);
    exportgraphics(trajectoryFigure, saved_files.png, 'Resolution', 300);

    fprintf('轨迹 FIG：%s\n', saved_files.figure);
    fprintf('轨迹 PNG：%s\n', saved_files.png);
end

% 如果正式阶段发生异常，由于本次不保存不完整数据，停车后直接重新抛出异常。
if ~isempty(formal_error)
    rethrow(formal_error);
end

%% ==================== 局部函数 ====================
function reached = moveToDesiredStartAligned( ...
    pub, sub, x_goal, y_goal, beta_goal, ...
    alignment_distance, alignment_position_tolerance, ...
    start_position_tolerance, start_yaw_tolerance, max_time)
% 适用于不能可靠原地旋转的小车的两阶段归位控制：
% 阶段1：以小速度圆弧转向并驶向预对准点。
% 阶段2：从预对准点沿圆弧/直线驶向期望起点，同时校正最终航向。
%
% 关键处理：
% 1. 转向阶段不再令 vc=0，避免底盘因无法原地旋转而永久卡住。
% 2. 归位角速度方向与正式控制器保持一致，使用标准符号映射。
%    先前 V9 中的整体反号会导致车辆向目标航向的反方向转动。
% 3. 到达起点但航向不合格时，返回预对准阶段重新接近，
%    不再依赖原地调头。

    global stopSimulation

    reached = false;

    align_x = x_goal - alignment_distance*cos(beta_goal);
    align_y = y_goal - alignment_distance*sin(beta_goal);

    %% 归位控制参数
    k_rho_align = 0.75;
    k_rho_final = 0.55;
    k_turn = 1.5;
    k_drive_heading = 1.2;

    v_max_align = 0.20;
    v_max_final = 0.20;
    w_max = 1.80;
    v_min_move = 0.15;
    slow_radius = 0.28;

    % 归位角速度方向与正式控制器保持一致。
    % heading_error<0 时发送负角速度，heading_error>0 时发送正角速度。
    homing_turn_sign = 1;

    % 滞环阈值仅用于切换“低速大转弯”和“正常驶向目标”。
    turn_enter = 25*pi/180;
    turn_exit = 8*pi/180;

    required_align_samples = 3;
    required_start_samples = 6;
    align_samples = 0;
    start_samples = 0;

    stage = 1;
    turning_mode = true;

    beta_previous = [];
    start_timer = tic;
    print_timer = tic;

    fprintf('预对准点：x = %.4f m, y = %.4f m\n', align_x, align_y);
    fprintf('归位角速度符号 homing_turn_sign = %d（与正式控制器一致）\n', homing_turn_sign);

    while toc(start_timer) < max_time
        drawnow;

        if stopSimulation
            stopRobot(pub);
            return;
        end

        [msg, ok] = receiveMocapWithStop(sub, 2);
        if ~ok
            stopRobot(pub);
            if stopSimulation
                return;
            end
            warning('归位阶段暂未收到动捕数据，继续等待。');
            continue;
        end

        x_current = msg.Pose.Position.X;
        y_current = msg.Pose.Position.Y;
        yaw_raw = msg.Pose.Position.Z;

        if isempty(beta_previous)
            beta_current = yaw_raw;
        else
            delta_yaw = wrapToPi(yaw_raw - beta_previous);
            beta_current = beta_previous + delta_yaw;
        end
        beta_previous = beta_current;

        beta_error = wrapToPi(beta_goal - beta_current);

        if stage == 1
            %% 阶段1：以非零线速度驶向预对准点
            dx = align_x - x_current;
            dy = align_y - y_current;
            distance_error = hypot(dx, dy);
            target_heading = atan2(dy, dx);
            heading_error_control = wrapToPi(target_heading - beta_current);

            if distance_error <= alignment_position_tolerance
                vc = 0;
                wc = 0;
                align_samples = align_samples + 1;
            else
                align_samples = 0;

                if turning_mode
                    if abs(heading_error_control) <= turn_exit
                        turning_mode = false;
                    end
                elseif abs(heading_error_control) >= turn_enter
                    turning_mode = true;
                end

                if turning_mode
                    % 不能原地旋转，因此保持最低线速度走小半径圆弧。
                    vc = v_min_move;
                    wc = homing_turn_sign*k_turn*heading_error_control;
                else
                    speed_magnitude = min(k_rho_align*distance_error, v_max_align);
                    speed_magnitude = max(speed_magnitude, v_min_move);

                    if distance_error < slow_radius
                        speed_magnitude = max( ...
                            v_min_move, ...
                            speed_magnitude*max(0.60, distance_error/slow_radius));
                    end

                    heading_scale = max(0.45, cos(abs(heading_error_control)));
                    vc = max(v_min_move, speed_magnitude*heading_scale);
                    wc = homing_turn_sign*k_drive_heading*heading_error_control;
                end
            end

            stage_text = '阶段1：前往预对准点';
            printed_heading_error = heading_error_control;

            if align_samples >= required_align_samples
                stage = 2;
                align_samples = 0;
                start_samples = 0;
                turning_mode = true;
                stopRobot(pub);

                fprintf(['\n已进入预对准点容差范围，', ...
                    '开始前往期望起点并校正最终航向。\n']);
                pauseWithStop(0.3);
                continue;
            end

        else
            %% 阶段2：驶向起点并通过接近方向校正最终航向
            dx = x_goal - x_current;
            dy = y_goal - y_current;
            distance_error = hypot(dx, dy);
            target_heading = atan2(dy, dx);
            target_heading_error = wrapToPi(target_heading - beta_current);
            beta_error = wrapToPi(beta_goal - beta_current);

            position_ok = distance_error <= start_position_tolerance;
            yaw_ok = abs(beta_error) <= start_yaw_tolerance;

            if position_ok && yaw_ok
                vc = 0;
                wc = 0;
                start_samples = start_samples + 1;
            elseif position_ok && ~yaw_ok
                % 底盘不能可靠原地旋转，退回预对准阶段重新直线接近。
                fprintf(['\n已到起点附近，但航向误差为 %.2f deg。', ...
                    '返回预对准阶段重新接近。\n'], beta_error*180/pi);
                stage = 1;
                turning_mode = true;
                align_samples = 0;
                start_samples = 0;
                stopRobot(pub);
                pauseWithStop(0.3);
                continue;
            else
                start_samples = 0;

                % 越接近起点，越强调最终期望航向 beta_goal。
                blend = (slow_radius - distance_error) / ...
                    max(slow_radius - start_position_tolerance, eps);
                blend = min(max(blend, 0), 0.75);

                heading_error_control = wrapToPi( ...
                    target_heading_error + ...
                    blend*wrapToPi(beta_error - target_heading_error));

                if turning_mode
                    if abs(heading_error_control) <= turn_exit
                        turning_mode = false;
                    end
                elseif abs(heading_error_control) >= turn_enter
                    turning_mode = true;
                end

                if turning_mode
                    vc = v_min_move;
                    wc = homing_turn_sign*k_turn*heading_error_control;
                else
                    speed_magnitude = min(k_rho_final*distance_error, v_max_final);
                    speed_magnitude = max(speed_magnitude, v_min_move);

                    if distance_error < slow_radius
                        speed_magnitude = max( ...
                            v_min_move, ...
                            speed_magnitude*max(0.60, distance_error/slow_radius));
                    end

                    heading_scale = max(0.50, cos(abs(heading_error_control)));
                    vc = max(v_min_move, speed_magnitude*heading_scale);
                    wc = homing_turn_sign*k_drive_heading*heading_error_control;
                end
            end

            stage_text = '阶段2：接近起点并校正朝向';
            printed_heading_error = beta_error;

            if start_samples >= required_start_samples
                reached = true;
                stopRobot(pub);

                fprintf('\n起点位置及朝向归位完成：\n');
                fprintf('实际位置：x = %.4f m, y = %.4f m\n', ...
                    x_current, y_current);
                fprintf('位置误差：%.4f m\n', distance_error);
                fprintf('实际航向：%.2f deg\n', beta_current*180/pi);
                fprintf('期望航向：%.2f deg\n', beta_goal*180/pi);
                fprintf('航向误差：%.2f deg\n', beta_error*180/pi);
                return;
            end
        end

        if stage == 1
            vc = max(min(vc, v_max_align), -v_max_align);
        else
            vc = max(min(vc, v_max_final), -v_max_final);
        end
        wc = max(min(wc, w_max), -w_max);

        drawnow;
        if stopSimulation
            stopRobot(pub);
            return;
        end

        sendVelocity(pub, vc, wc);

        if toc(print_timer) >= 0.25
            if stage == 1
                fprintf(['%s | x = %.3f, y = %.3f | ', ...
                    '距离误差 = %.3f m | 目标点航向误差 = %.2f deg | ', ...
                    '最终航向误差 = %.2f deg | v = %.3f, w = %.3f\n'], ...
                    stage_text, x_current, y_current, ...
                    distance_error, printed_heading_error*180/pi, ...
                    beta_error*180/pi, vc, wc);
            else
                fprintf(['%s | x = %.3f, y = %.3f | ', ...
                    '距离误差 = %.3f m | 最终航向误差 = %.2f deg | ', ...
                    'v = %.3f, w = %.3f\n'], ...
                    stage_text, x_current, y_current, ...
                    distance_error, printed_heading_error*180/pi, ...
                    vc, wc);
            end
            print_timer = tic;
        end

        pauseWithStop(0.02);
    end

    stopRobot(pub);
    warning('起点位置及朝向归位超时，最大允许时间为 %.1f s。', max_time);
end

function waitForRunConfirmation()
% 归位完成后保持停车，等待 Space 或 Enter。
% 等待期间 S 键仍可随时紧急停车。

    global stopSimulation runConfirmed

    while ~runConfirmed && ~stopSimulation
        drawnow;
        pause(0.02);
    end
end

function [msg, ok] = receiveMocapWithStop(sub, total_timeout)
% 短周期轮询接收动捕，使等待数据期间 S 键仍可被处理。

    global stopSimulation

    msg = [];
    ok = false;
    timer_start = tic;

    while toc(timer_start) < total_timeout
        drawnow;

        if stopSimulation
            return;
        end

        try
            msg = receive(sub, 0.2);
            ok = true;
            return;
        catch
            % 单次超时后继续轮询
        end
    end
end

function pauseWithStop(duration_seconds)
% 将长 pause 拆成短间隔，使 S 键事件持续有效。

    global stopSimulation

    timer_start = tic;
    while toc(timer_start) < duration_seconds
        drawnow;

        if stopSimulation
            return;
        end

        remaining_time = duration_seconds - toc(timer_start);
        if remaining_time <= 0
            return;
        end

        pause(min(0.02, remaining_time));
    end
end

function onKeyPressed(event)
% S：全程紧急停车。
% Space/Enter：仅在等待二次确认时允许开始正式实验。

    global stopSimulation cmdpub runConfirmed waitingForConfirmation

    if strcmpi(event.Key, 's')
        stopSimulation = true;
        runConfirmed = false;
        stopRobot(cmdpub);
        fprintf('\n*** 已按下 S 键，小车紧急停车。***\n');
        return;
    end

    is_confirm_key = strcmpi(event.Key, 'space') ...
        || strcmpi(event.Key, 'return') ...
        || strcmpi(event.Key, 'enter');

    if is_confirm_key && waitingForConfirmation && ~stopSimulation
        runConfirmed = true;
        stopRobot(cmdpub);
        fprintf('\n*** 已完成二次确认，准备开始正式实验。***\n');
    end
end

function onControlFigureClosed(src)
% 关闭按键监听窗口时同样紧急停车。

    global stopSimulation cmdpub runConfirmed waitingForConfirmation

    stopSimulation = true;
    runConfirmed = false;
    waitingForConfirmation = false;
    stopRobot(cmdpub);

    if isgraphics(src)
        delete(src);
    end

    fprintf('\n*** 控制窗口已关闭，小车紧急停车。***\n');
end

function sendVelocity(pub, vc, wc)
% 发送线速度和角速度。

    if isempty(pub)
        return;
    end

    msg = rosmessage(pub);
    msg.Linear.X = vc;
    msg.Angular.Z = wc;
    send(pub, msg);
end

function stopRobot(pub)
% 发送零速度。可安全重复调用。

    if isempty(pub)
        return;
    end

    try
        sendVelocity(pub, 0, 0);
    catch stopError
        warning('发送停车指令失败：%s', stopError.message);
    end
end

function [vc, wc] = limiter(vc, wc)
% 正式控制阶段速度限幅。

    vc_min = -0.65;
    vc_max = 0.65;
    wc_min = -1.5;
    wc_max = 1.5;

    vc = max(min(vc, vc_max), vc_min);
    wc = max(min(wc, wc_max), wc_min);
end
