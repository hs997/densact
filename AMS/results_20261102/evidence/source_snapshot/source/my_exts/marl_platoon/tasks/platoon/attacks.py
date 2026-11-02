import torch


def sample_hybrid_attack(num_envs: int, device: torch.device, max_fdi_pos: float, max_fdi_acc: float, max_dos_rate: float):
    """Legacy non-HAPPO hybrid DoS+FDI sampler.

    返回：
      beta_a, beta_p: {0,1} 掉包开关
      f_a, f_p:       注入扰动

    HAPPO/MGRS 任务使用 task-local CA-GAN attacker；该函数仅服务旧 wrapper 日志路径。
    """
    # DoS: 1=正常, 0=丢包
    keep_prob = 1.0 - max_dos_rate
    beta_a = (torch.rand(num_envs, device=device) < keep_prob).float()
    beta_p = (torch.rand(num_envs, device=device) < keep_prob).float()

    # FDI 幅值受约束
    f_a = (2.0 * torch.rand(num_envs, device=device) - 1.0) * max_fdi_acc
    f_p = (2.0 * torch.rand(num_envs, device=device) - 1.0) * max_fdi_pos

    return beta_a, beta_p, f_a, f_p
