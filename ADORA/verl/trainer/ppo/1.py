def compute_grpo_outcome_advantage_length_weight_v2(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    correct_score: float = 1.01,
    beta: float = 0.5,
    p: float = 0.5,
    lamda: float = 0.1
):
    """
    最终修复版本，解决数据类型问题并优化性能
    """
    # 基础参数准备
    device = token_level_rewards.device
    response_length_dim = token_level_rewards.shape[-1]
    scores = (token_level_rewards * (token_level_rewards != 0)).sum(dim=-1)
    response_lengths = eos_mask.sum(dim=1).float()
    original_scores = scores.clone()

    # 分组数据结构
    id2original = defaultdict(list)
    id2length = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 1. 收集分组信息（优化为批量处理）
        for i in range(bsz):
            idx = str(index[i])
            id2original[idx].append(original_scores[i].item())
            id2length[idx].append(response_lengths[i].item())

        # 2. 预处理分组参数（使用浮点张量）
        group_params = {}
        for idx in id2original:
            # 转换为浮点张量
            raw_scores = torch.tensor(id2original[idx], dtype=torch.float32, device=device)
            lengths = torch.tensor(id2length[idx], dtype=torch.float32, device=device)
            
            pos_mask = raw_scores > correct_score
            group_params[idx] = {
                'n_samples': len(raw_scores),
                'pos_count': pos_mask.sum().item(),
                'pos_lengths': lengths[pos_mask],
                'neg_lengths': lengths[~pos_mask]
            }

        # 3. 动态调整奖励（向量化处理）
        adjusted_scores = []
        for i in range(bsz):
            idx = str(index[i])
            params = group_params[idx]
            raw_score = original_scores[i].item()
            length = response_lengths[i].item()
            
            if raw_score > correct_score:
                pos_count = params['pos_count']
                n_total = params['n_samples']
                pos_lengths = params['pos_lengths']

                if pos_count < n_total * p and len(pos_lengths) > 0:  # 难题
                    max_len = pos_lengths.max()
                    min_len = pos_lengths.min()
                    len_range = torch.clamp(max_len - min_len, min=1e-6)
                    norm_len = (length - min_len) / len_range
                    adjusted = raw_score + beta * (2 * norm_len.item() - 1)
                elif pos_count >= n_total * (1 - p) and len(pos_lengths) > 0:  # 简单题
                    max_len = pos_lengths.max()
                    min_len = pos_lengths.min()
                    len_range = torch.clamp(max_len - min_len, min=1e-6)
                    norm_len = (length - min_len) / len_range
                    adjusted = raw_score - beta * (2 * norm_len.item() - 1)
                else:
                    adjusted = raw_score
            else:
                adjusted = raw_score
            adjusted_scores.append(adjusted)

        scores = torch.tensor(adjusted_scores, device=device, dtype=torch.float32)

        # 4. 分组标准化（优化计算）
        unique_indices = list(group_params.keys())
        for idx in unique_indices:
            mask = torch.tensor([str(index[i]) == idx for i in range(bsz)], device=device)
            group_scores = scores[mask]
            
            if group_scores.numel() > 1:
                mean = group_scores.mean()
                std = group_scores.std()
            else:
                mean, std = 0.0, 1.0
            
            # 应用标准化
            scores[mask] = (group_scores - mean) / (std + epsilon)
            
            # 应用权重
            params = group_params[idx]
            if len(params['neg_lengths']) > 0 and len(params['pos_lengths']) > 0:
                mean_neg = params['neg_lengths'].mean()
                max_pos = params['pos_lengths'].max()
                if not (max_pos > mean_neg):
                    scores[mask] *= lamda

        # 5. 扩展为序列格式
        scores = scores.unsqueeze(-1).expand(-1, response_length_dim) * eos_mask

    return scores, scores