model="xxx" #model path
config="vllm"
num_limit=100
max_token=8192
ntrain=0
split="test"
log_path="log/stage1_1000step"

mkdir -p ${log_path}

for eval_nppl in 2 3 4 5 6 7 8; do
    log_file="${log_path}/${eval_nppl}.log"
    echo "Starting job for eval_nppl: $eval_nppl, logging to $log_file"

    CUDA_VISIBLE_DEVICES=$((eval_nppl - 1)) PYTHONUNBUFFERED=1 python main_eval_instruct.py --batch_size 8 --model ${model} --max_token ${max_token} \
    --ntrain ${ntrain} --config ${config} --limit ${num_limit} --split ${split} --temperature 1.0  --top_p 1.0 \
    --problem_type "clean" --eval_nppl ${eval_nppl} > "$log_file" 2>&1 &
done &  