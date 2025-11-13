# DPOloss
import torch
import torch.nn.functional as F


def dpo_loss(pi_w, pi_l, beta=0.1):
    """
    DPO 核心损失（变量极简版）
    输入：
        pi_w: 新策略对「赢样本w」的log概率 (batch_size,) —— 人类偏好的回复
        pi_l: 新策略对「输样本l」的log概率 (batch_size,) —— 人类不偏好的回复
        beta: 温度超参（控制偏好强度，默认0.1，工业界常用）
    输出：
        dpo_total_loss: DPO总损失
    """
    # 1. 计算偏好加权后的log概率（beta放大偏好差异）
    w_score = beta * pi_w
    l_score = beta * pi_l
    # 对比损失 让偏好样本相对概率最大化（交叉熵形式）
    # 维度堆叠 不同于cat
    # 等价于 -log( exp(w_score) / (exp(w_score) + exp(l_score)) )
    logits = torch.stack([w_score, l_score], dim=1)
    loss = -torch.mean(F.log_softmax(logits, dim=1)[:, 0])  # 只关注样本

    # 熵正则
    entorpy = -torch.mean(pi_w)
    entorpy_loss = -c2 * entorpy
    total_loss = entorpy_loss + loss
    return total_loss


if __name__ == "__mian__":
    # 模拟batch_size=32的偏好数据（赢/输样本的log概率）
    batch_size = 32
    pi_w = torch.randn(batch_size)  # 赢样本log概率
    pi_l = torch.randn(batch_size)  # 输样本log概率

    loss = dpo_loss(pi_w, pi_l)
    print("DPO损失:", loss.item())
