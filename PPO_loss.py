import torch
import torch.nn as nn
import torch.nn.functional as F


def gae_advantage(v_now, v_next, r, gamma: float = 0.99, gae_lambda: float = 0.95):
    """
    计算GAE（广义优势估计），单独抽离函数，逻辑更清晰
    输入：
        v_now: 当前状态价值 (batch_size,) 或 (traj_len,)
        v_next: 下一个状态价值 (batch_size,) 或 (traj_len,)
        r: 即时奖励 (batch_size,) 或 (traj_len,)
        gamma: 折扣因子（与主函数一致）
        gae_lambda: GAE参数（默认0.95，平衡偏差-方差）
    输出：
        A: GAE优势值 (batch_size,) 或 (traj_len,)
    """
    # 计算时序差分误差 td-error
    td_error = r + gamma * v_next - v_now

    # 反向累加计算GAE从最后时间步倒推
    A = torch.zeros_like(td_error)
    advantage = 0.0  # 最后一个时间步无后续为0

    # 反向遍历 从后往前  (γλ)^k * δ_{t+k}
    for t in reversed(range(len(td_error))):
        advantage = td_error[t] + gamma * gae_lambda * advantage
        A[t] = advantage
    # 优势标准
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A


def ppo_loss(
    pi_new,
    pi_old,
    v_now,
    r,
    v_next,
    eps=0.2,
    c1=0.2,
    c2=0.01,
    gamma=0.99,
    gae_lambda=0.95,
):
    """
    PPO核心损失计算（GAE优势值版，变量极简+逻辑清晰）
    输入新增：
        gae_lambda: GAE参数（默认0.95，工业界常用值）
    其他输入/输出与原版本一致
    """
    # 计算gae
    A = gae_advantage(v_now, v_next, r, gamma, gae_lambda)
    # 策略损失  pi_new = log(prob_new)、pi_old = log(prob_old)
    ratio = torch.exp(pi_new - pi_old)
    clip_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
    suur1 = ratio * A
    suur2 = clip_ratio * A
    policy_loss = -torch.mean(torch.min(suur1, suur2))

    # 价值损失
    td_target = r + gamma * v_next
    value_loss = c1 * F.mse_loss(v_now, td_target)
    # 熵正则 鼓励策略探索
    entropy = -torch.mean(pi_new)
    entropy_loss = -c2 * entropy
    total_loss = policy_loss + value_loss + entropy_loss
    return total_loss


if __name__ == "__main__":
    traj_len = 10
    pi_new = torch.randn(traj_len)  # 新策略log概率
    pi_old = torch.randn(traj_len)  # 旧策略log概率
    v_now = torch.randn(traj_len)  # 当前状态价值
    r = torch.randn(traj_len)  # 即时奖励
    v_next = torch.randn(traj_len)  # 下一个状态价值（最后一步可设为0，因为无后续状态）

    # 计算损失
    loss = ppo_loss(pi_new, pi_old, v_now, r, v_next)
    print("PPO总损失（GAE版）:", loss.item())
