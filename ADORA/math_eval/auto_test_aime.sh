CHECKPOINT_PATH="$1"/actor
mkdir -p /volume/ailab4sci/ztgao/log/"$2"
mkdir -p /volume/ailab4sci/ztgao/aime/"$2"

if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo "提供的路径无效: $CHECKPOINT_PATH"
  exit 1
fi

CHECKPOINTS=($(find "$CHECKPOINT_PATH" -mindepth 1 -maxdepth 1 -type d))

declare -A GPU_LAST_USED_TIME
declare -a CHECKPOINT_QUEUE

check_gpu_free() {
    local gpu_id=$1
    local pid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu_id")
    local current_time=$(date +%s)

    if [ -n "$pid" ]; then
        GPU_LAST_USED_TIME["$gpu_id"]=$current_time
        return 1
    fi

    if [ -z "${GPU_LAST_USED_TIME["$gpu_id"]}" ] || [ $((current_time - GPU_LAST_USED_TIME["$gpu_id"])) -gt 600 ]; then
        GPU_LAST_USED_TIME["$gpu_id"]=$current_time
        return 0
    else
        return 1
    fi
}

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    step=$(basename "$CHECKPOINT" | sed -E 's/[^0-9]*([0-9]+)$/\1/')
    log_path="/volume/ailab4sci/ztgao/log/$2/$step.log"

    if [ ! -f "$log_path" ]; then
        CHECKPOINT_QUEUE+=("$CHECKPOINT")
        echo "已添加 checkpoint $CHECKPOINT 到队列。"
    else
        echo "日志文件已存在: $log_path, 跳过此模型。"
    fi
done

while [ ${#CHECKPOINT_QUEUE[@]} -gt 0 ]; do
    CHECKPOINT=${CHECKPOINT_QUEUE[0]}
    step=$(basename "$CHECKPOINT" | sed -E 's/[^0-9]*([0-9]+)$/\1/')
    log_path="/volume/ailab4sci/ztgao/log/$2/$step.log"

    GPU_ID=""
    for i in $(seq 0 7); do
        if check_gpu_free "$i"; then
            GPU_ID=$i
            break
        fi
    done

    if [ -z "$GPU_ID" ]; then
        echo "没有空闲的 GPU，等待 30 秒后重试..."
        sleep 30
        continue
    fi

    json_path="/volume/ailab4sci/ztgao/aime/$2/$step.json"
    mkdir -p "$(dirname "$log_path")"
    echo "使用 GPU $GPU_ID 处理模型: $CHECKPOINT, log: $log_path"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 test_aime.py --model_path "$CHECKPOINT" --json_path "$json_path" > "$log_path" 2>&1 &
    CHECKPOINT_QUEUE=("${CHECKPOINT_QUEUE[@]:1}")
    sleep 2
done

wait
echo "所有评估任务已完成。"
