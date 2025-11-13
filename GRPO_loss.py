import torch
import torch.nn.functional as F


def grpo_loss(pi_new, pi_old, r, group_size=4, eps=0.2, c2=0.01):
    """
    GRPO 核心损失（变量极简版，无Critic，组内对比估算优势）
    输入：
        pi_new: 新策略的log概率 (batch_size × group_size,) —— 组内所有样本的logp
        pi_old: 旧策略的log概率 (batch_size × group_size,) —— 固定不更新
        r: 组内样本的奖励/偏好分数 (batch_size × group_size,) —— 组内每条回复的得分
        group_size: 每组采样数量（默认4，常用值）
        eps: 裁剪系数（同PPO，默认0.2）
        c2: 熵正则权重（默认0.01）
    输出：
        grpo_total_loss: GRPO总损失
    """
    # pi_new.shape[0] ->[bsz,c,w,h]->bsz批次大小样本总数
    # 1. 重构维度：适应组内计算 (batch_size, group_size)
    batch_size = pi_new.shape[0] // group_size
    pi_new = pi_new.reshape(batch_size, group_size)
    pi_old = pi_old.reshape(batch_size, group_size)  # [bsz*group_size=pi_neww.shape[0]]
    r = r.reshape(batch_size, group_size)

    # 组内优势计算 A=（样本奖励-组内平均奖励）/std(A)+eps
    r_mean = r.mean(dim=1, keepdim=True)
    r_std = r.std(dim=1, keepdim=True)
    A = (r - r_mean) / (r_std + 1e-8)

    # 策略损失
    radio = torch.exp(pi_new - pi_old)
    radio_clamp = torch.clamp(radio, 1 - eps, 1 + eps)
    surr1 = radio * A
    surr2 = radio_clamp * A
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    # 熵正则
    entropy = -torch.mean(pi_new, dim=1)
    entropy_loss = -c2 * entropy.mean()
    total_loss = policy_loss + entropy_loss
    return total_loss


if __name__ == "__main__":
    # 模拟输入：batch_size=8组，每组4个样本（total=32个样本）
    batch_size = 8
    group_size = 4
    total_samples = batch_size * group_size

    pi_new = torch.randn(total_samples)  # 新策略log概率
    pi_old = torch.randn(total_samples)  # 旧策略log概率（固定）
    r = torch.randn(total_samples)  # 组内样本奖励（可正可负）

    loss = grpo_loss(pi_new, pi_old, r, group_size=group_size)
    print("带标准差归一化的GRPO总损失:", loss.item())
