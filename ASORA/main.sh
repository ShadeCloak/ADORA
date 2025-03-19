export N_GPUS=4
export BASE_MODEL=/mnt/hwfile/trustai/renqingnan/Qwen/Qwen2.5-Math-7B
export DATA_DIR=/mnt/petrelfs/renqingnan/xietian/Logic-RL/kk_data/3ppl_train
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=kk-Qwen2.5-Math-7B
export VLLM_ATTENTION_BACKEND=XFORMERS

srun -p AI4Good_L -t 24:00:00 --gres=gpu:$N_GPUS bash ./scripts/train_ppo.sh

# actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \

srun -p AI4Good_L -t 24:00:00 --gres=gpu:$N_GPUS bash ./scripts/train_ppo.sh

# actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \