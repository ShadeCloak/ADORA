# 
pip install -e ADORA_VL --no-deps 

export OPENRLHF_PATH='ADORA_VL'
export PYTHONPATH=$OPENRLHF_PATH:$PYTHONPATH

ray stop --force
ray start --head \
    --node-ip-address=0.0.0.0 \
    --num-gpus=8

ray job submit --address=http://127.0.0.1:8265 \
    -- python -m openrlhf.cli.train_ppo_ray_vl \
    --save_steps 50 \
    --save_hf_ckpt \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
    --colocate_actor_ref \
    --vllm_sync_backend nccl \
    --pretrain Qwen2.5-VL-7B-Instruct \
    --save_path Qwen2.5-VL-7B-Instruct-ADORA \
    --prompt_data ../geometry3k \
    --remote_rm_url examples/scripts/reward_func_math.py \
    --input_key problem \
    --images_key images \
    --reference_key answer \
    --apply_chat_template \
    --system_prompt 'Please reason step by step, and put your final answer within \boxed{}.'\
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 128 \
    --max_epochs 1 \
    --num_episodes 15 \
    --prompt_max_len 4096 \
    --generate_max_len 4096 \
    --zero_stage 3 \
    --bf16 \
    --gradient_checkpointing \
    --actor_learning_rate 1.0e-6 \
    --lr_warmup_ratio 0 \
    --l2 1.0e-2 \
    --init_kl_coef 0.001 \
    --use_kl_loss \
    --kl_estimator k3 \
    --use_adora \
    --adam_offload \
    --flash_attn \
    --n_samples_per_prompt 8 \
    --advantage_estimator group_norm 

