
# ADORA_LLM

## Installation

```bash
conda create -n adora python=3.9
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib
```

## Data Preparation

You can directly use /data.

For your own data generation, here's a demo:

### Base Model
```bash
python ./examples/data_preprocess/kk.py \
    --local_dir {processed_data_path} \
    --data_path {raw_data_path}
```

### Instruct Model
```bash
python ./examples/data_preprocess/kk.py \
    --template_type=qwen-instruct \
    --local_dir {processed_data_path} \
    --data_path {raw_data_path}
```

---

## Training Execution
```bash
conda activate adora
bash ADORA.sh  # 4×A100 80G
```

---

## ⚙️ Implementation Details

| Component              | Location                          |
|------------------------|-----------------------------------|
| Hyperparameter setting | `ADORA/verl/trainer/config/ppo_trainer.yaml` |
| Enter ADORA | `ADORA/verl/trainer/ppo/ray_trainer.py` |
| ADORA strategy | `ADORA/verl/trainer/ppo/core_algos.py` |

## Weight Function

In the blog, compute_grpo_outcome_advantage_ADORA is defined as follows, feel free to modify it according to your own scenario.

```python
def compute_grpo_outcome_advantage_ADORA(
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
    
    scores = (token_level_rewards * (token_level_rewards != 0)).sum(dim=-1)
    response_lengths = eos_mask.sum(dim=1).float()
    original_scores = scores.clone()

    id2original = defaultdict(list)
    id2length = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        
        for i in range(bsz):
            idx = str(index[i])
            id2original[idx].append(original_scores[i].item())
            id2length[idx].append(response_lengths[i].item())

        group_params = {}
        for idx in id2original:
            raw_scores = torch.tensor(id2original[idx], dtype=torch.float32, device=device)
            lengths = torch.tensor(id2length[idx], dtype=torch.float32, device=device)
            
            pos_mask = raw_scores > correct_score
            neg_mask = (raw_scores <= correct_score) & (lengths < max_length_threshold)
            
            pos_lengths = lengths[pos_mask]
            neg_lengths = lengths[neg_mask]
            
            max_pos = pos_lengths.max().item() if pos_lengths.numel() > 0 else -float('inf')
            mean_neg = neg_lengths.mean().item() if neg_lengths.numel() > 0 else -float('inf')
            
            group_params[idx] = {
                'n_samples': len(raw_scores),
                'pos_count': pos_mask.sum().item(),
                'pos_lengths': pos_lengths,
                'neg_lengths': neg_lengths,
                'max_pos': max_pos,
                'mean_neg': mean_neg
            }

        unique_indices = list(group_params.keys())
        for idx in unique_indices:
            mask = torch.tensor([str(index[i]) == idx for i in range(bsz)], device=device)
            group_scores = scores[mask]
            
            if group_scores.numel() > 1:
                mean = group_scores.mean()
                std = group_scores.std()
                scores[mask] = (group_scores - mean) / (std + epsilon)
            else:
                scores[mask] = scores[mask]
                
        for idx in unique_indices:
            params = group_params[idx]
            if not params['length_advantage']:
                mask = torch.tensor([str(index[i]) == idx for i in range(bsz)], device=device)
                scores[mask] *= lamda

        scores = scores.unsqueeze(-1).expand(-1, response_length_dim) * eos_mask

    return scores, scores
```

