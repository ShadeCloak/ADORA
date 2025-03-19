def compute_grpo_outcome_advantage_length_weight_v2(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    correct_score: float = 1.01,
    beta: float = 0.5,
    p: float = 0.5,
    lamda: float = 0.1,
    max_length_threshold: int = 4054
):
    """
    完善版GRPO优势计算，修正动态调整与长度优势的逻辑关系
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

        # 5. 动态奖励调整（仅在长度占优时进行）
        adjusted_scores = []
        for i in range(bsz):
            idx = str(index[i])
            params = group_params[idx]
            raw_score = original_scores[i].item()
            length = response_lengths[i].item()
            
            # 仅当满足长度优势条件时调整
            if raw_score > correct_score and params.get('length_advantage', False):
                pos_count = params['pos_count']
                n_total = params['n_samples']
                pos_lengths = params['pos_lengths']
                
                if pos_lengths.numel() == 0:
                    adjusted = raw_score
                else:
                    # 根据样本分布动态调整
                    if pos_count < n_total * p:  # 困难模式
                        max_len = pos_lengths.max()
                        min_len = pos_lengths.min()
                        len_range = torch.clamp(max_len - min_len, min=1e-6)
                        norm_len = (length - min_len) / len_range
                        adjusted = raw_score + beta * (2 * norm_len.item() - 1)
                    elif pos_count >= n_total * (1 - p):  # 简单模式
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

        # 6. 分组标准化
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

        # 7. 自适应权重衰减（长度不占优时应用）
        for idx in unique_indices:
            params = group_params[idx]
            if not params['length_advantage']:
                mask = torch.tensor([str(index[i]) == idx for i in range(bsz)], device=device)
                scores[mask] *= lamda

        # 8. 序列格式扩展
        scores = scores.unsqueeze(-1).expand(-1, response_length_dim) * eos_mask

    return scores, scores