import torch
import torch.nn as nn


class RewardTeacher(nn.Module):
    """简化版 Reward Teacher。

    作用：提供 F_phi(obs, action) 作为 shaped reward 增量。
    备注：这是“可运行骨架版”，先打通训练与日志链路，后续可替换为完整 meta-gradient 更新。
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # obs: [B, obs_dim], action: [B, act_dim]
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class TeacherState:
    """训练期 Teacher 状态容器（占位）。"""

    def __init__(self):
        self.enabled = True
        self.last_meta_loss = 0.0
