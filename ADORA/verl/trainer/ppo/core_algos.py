# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def compute_grpo_outcome_advantage_ASORA(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    correct_score: float = 1.01,
    lamda: float = 0.1,
    max_length_threshold: int = 4054
):
    """

    """
    device = token_level_rewards.device
    response_length_dim = token_level_rewards.shape[-1]
    
    # 1. 基础参数计算
    scores = (token_level_rewards * (token_level_rewards != 0)).sum(dim=-1)
    response_lengths = eos_mask.sum(dim=1).float()
    original_scores = scores.clone()

    # 2. 分组数据结构初始化
    id2original = defaultdict(list)
    id2length = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 3. 收集分组信息
        for i in range(bsz):
            idx = str(index[i])
            id2original[idx].append(original_scores[i].item())
            id2length[idx].append(response_lengths[i].item())

        # 4. 预处理分组参数（含长度优势预计算）
        group_params = {}
        for idx in id2original:
            raw_scores = torch.tensor(id2original[idx], dtype=torch.float32, device=device)
            lengths = torch.tensor(id2length[idx], dtype=torch.float32, device=device)
            
            # 正负样本划分
            pos_mask = raw_scores > correct_score
            neg_mask = (raw_scores <= correct_score) & (lengths < max_length_threshold)
            
            # 关键参数计算
            pos_lengths = lengths[pos_mask]
            neg_lengths = lengths[neg_mask]
            
            # 长度优势判断
            max_pos = pos_lengths.max().item() if pos_lengths.numel() > 0 else -float('inf')
            mean_neg = neg_lengths.mean().item() if neg_lengths.numel() > 0 else -float('inf')
            
            group_params[idx] = {
                'n_samples': len(raw_scores),
                'pos_count': pos_mask.sum().item(),
                'pos_lengths': pos_lengths,
                'neg_lengths': neg_lengths,
                'length_advantage': (max_pos > mean_neg),  # 核心判断标志
                'max_pos': max_pos,
                'mean_neg': mean_neg
            }

        # 5. 分组标准化
        unique_indices = list(group_params.keys())
        for idx in unique_indices:
            mask = torch.tensor([str(index[i]) == idx for i in range(bsz)], device=device)
            group_scores = scores[mask]
            
            if group_scores.numel() > 1:
                mean = group_scores.mean()
                std = group_scores.std()
                scores[mask] = (group_scores - mean) / (std + epsilon)
            else:
                scores[mask] = scores[mask]  # 单样本无需标准化

        # 6. 自适应权重衰减（长度不占优时应用）
        for idx in unique_indices:
            params = group_params[idx]
            if not params['length_advantage']:
                mask = torch.tensor([str(index[i]) == idx for i in range(bsz)], device=device)
                scores[mask] *= lamda

        # 7. 序列格式扩展
        scores = scores.unsqueeze(-1).expand(-1, response_length_dim) * eos_mask

    return scores, scores


def compute_grpo_outcome_advantage_length_weight(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    reward_threshold: float = 1.0,
    lamda: float = 0.1
):
    # 保持原始代码结构
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    
    # 新增分组长度统计
    response_lengths = eos_mask.sum(dim=-1)  # (bs,)
    
    id2score = defaultdict(list)
    id2length = defaultdict(list)  # 新增长度记录
    id2mean = {}
    id2std = {}
    id2weight = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 分组收集分数和长度
        for i in range(bsz):
            idx = index[i]  # 直接使用张量索引
            id2score[idx].append(scores[i])
            id2length[idx].append(response_lengths[i])  # 记录长度
            
        # 分组处理逻辑
        for idx in id2score:
            # 原始标准化逻辑
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
            
            # 新增权重计算逻辑
            group_scores = torch.stack(id2score[idx])
            group_lengths = torch.stack(id2length[idx])
            
            # 高低奖励划分
            high_mask = group_scores > reward_threshold
            low_mask = ~high_mask
            
            # 计算统计量
            max_high = group_lengths[high_mask].max() if high_mask.any() else -torch.inf
            mean_low = group_lengths[low_mask].mean() if low_mask.any() else torch.inf
            
            # 权重决策
            if (max_high > mean_low) or (not high_mask.any()):
                id2weight[idx] = 1.0
            else:
                id2weight[idx] = lamda

        # 应用权重和标准化
        for i in range(bsz):
            idx = index[i]
            scores[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)
            scores[i] *= id2weight[idx]  # 应用权重
            
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1] 
    non_zero_mask = (token_level_rewards != 0)  
 
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    # 用字典存储每个prompt id对应的分数列表
    id2score = defaultdict(list)  # Dict[str, List[float]]
    id2mean = {}  # Dict[str, float] 
    id2std = {}   # Dict[str, float]

    with torch.no_grad():
        bsz = scores.shape[0]  # batch_size
        
        # 按prompt id分组收集分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            
        # 计算每组内的均值和标准差
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 只有一个样本时,均值为0,标准差为1
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # 多个样本时,计算均值和标准差
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
                
        # 对每个样本进行标准化
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            
        # 将标准化后的分数扩展到序列长度维度,并用eos_mask遮盖
        # shape: (batch_size, response_length)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    # 返回标准化后的分数作为优势和回报
    # shape: (batch_size, response_length), (batch_size, response_length)
    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
