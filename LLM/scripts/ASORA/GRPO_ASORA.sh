set -x
# MODEL_PATH=/mnt/petrelfs/renqingnan/Qwen/Qwen2.5-7B-Instruct-1M
MODEL_PATH=/mnt/petrelfs/renqingnan/Qwen/Qwen2.5-7B-Instruct-1M
# 8->224 4->450 
export NAME=grpo_lr1e-6_v1_weight_beta0.126_p0.2_lamda0.1_length_6500_4

export WANDB_API_KEY=fe3a3f867639b3d57b39d7af7d0527150fc052fe
export RAY_TMPDIR=/tmp/ray/length_3900_4
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
DATA_PATH=/mnt/petrelfs/renqingnan/xietian/Logic-RL/kk_data


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    Length_weight.length_weight=True \
    Length_weight.correct_score=1.01 \
    Length_weight.lamda=0.1 \
    Length_weight.max_length_threshold=3900 \
    data.train_files=[${DATA_PATH}/3ppl_train_ins/train.parquet,${DATA_PATH}/4ppl_train_ins/train.parquet,${DATA_PATH}/5ppl_train_ins/train.parquet,${DATA_PATH}/6ppl_train_ins/train.parquet,${DATA_PATH}/7ppl_train_ins/train.parquet] \
    data.val_files=/mnt/petrelfs/renqingnan/xietian/Logic-RL/kk_data/test/ins/7ppl/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='reft-exp' \
    trainer.experiment_name=$NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=checkpoints/reft-exp/$NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=50 $@ 2>&1 | tee /mnt/petrelfs/renqingnan/xietian/Logic-RL/scripts/rqn/3_12/$NAME.log

ray stop
# srun -p AI4Good_S1 -t 240:00:00 --gres=gpu:8 bash /mnt/petrelfs/renqingnan/xietian/Logic-RL/scripts/rqn/3_12/baseline_length_4.sh