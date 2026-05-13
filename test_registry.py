import sys
import os

# 1. 强制指路到 IsaacLab 核心
sys.path.insert(0, r"E:\issca-sim\IsaacLab-main\source\isaaclab")

# 2. 尝试导入
try:
    import marl_platoon
    import gymnasium as gym
    print(f"✅ 包导入成功！")
    # 打印具体路径，确认是不是读到了 __init__.py
    print(f"   包的位置: {marl_platoon.__file__}")

    # 3. 检查任务注册
    target_task = "Isaac-Marl-Platoon-v0"
    if target_task in gym.envs.registry.keys():
        print(f"\n🎉 完美！任务 [{target_task}] 已注册！")
    else:
        print(f"\n❌ 失败：包导入了，但任务没注册。请检查 marl_platoon/__init__.py")

except ImportError as e:
    print(f"\n❌ 致命错误：根本找不到 marl_platoon 包。\n{e}")