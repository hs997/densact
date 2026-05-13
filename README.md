版本说明
framework_teacher_attack_v1

简述
在 IsaacLab 官方入口不变的前提下，完成了 marl_platoon 任务内部化训练框架接线：

HAPPO 作为 task-internal student 运行
teacher / attacker / shield 通过统一 pipeline 接入
支持 attack 强度分级、teacher 轻量 shaping、shield 前瞻防抖
当前版本已通过 easy / hard attack 与 teacher 稳定性验证，并已推送到 GitHub
