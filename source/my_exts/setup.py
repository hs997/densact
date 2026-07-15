from setuptools import setup, find_packages

setup(
    name="marl_platoon",
    version="0.1.0",
    description="My Multi-Agent RL Platoon Research Code",
    # 自动寻找当前目录下的所有包
    packages=find_packages(),
    # 这里声明依赖，以后你的算法缺什么包（比如 torch, gymnasium），都在这里加
    install_requires=[
        "isaaclab",  # 依赖核心库
    ],
)