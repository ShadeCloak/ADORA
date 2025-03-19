# Logic R1

## Installation

```
conda create -n logic python=3.9
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

pip install wandb IPython matplotlib
```

## Data Preparation

### For Base model
```
conda activate logic
python ./examples/data_preprocess/kk.py --local_dir {where to put the generated data} --data_path {where the origin kk train data is}
```
srun -p AI4Good_L -t 5 python ./examples/data_preprocess/kk.py --template_type=qwen-instruct --local_dir /mnt/petrelfs/renqingnan/xietian/Logic-RL/kk_data/test/ins --data_path /mnt/petrelfs/renqingnan/xietian/knights-and-knaves/test/people3_num100.jsonl

srun -p AI4Good_L -t 5 python ./examples/data_preprocess/kk.py --template_type=qwen-instruct --local_dir /mnt/petrelfs/renqingnan/xietian/Logic-RL/kk_data/8ppl_train_ins --data_path /mnt/petrelfs/renqingnan/xietian/knights-and-knaves/train/people8_num1000.jsonl

### For Instruct model
```
conda activate logic
python ./examples/data_preprocess/kk.py --template_type=qwen-instruct --local_dir {where to put the generated data} --data_path {where the origin kk train data is}
```

## Run Training
```
conda activate logic
bash main.sh
```

## Notes
- Reward score calculation can be found in `verl/utils/reward_score/kk.py`
- Real training script can be found in `scripts/train_ppo.sh`
- The model is trained with 4 GPUs in 1 node, you can change the number of GPUs in `kk.sh`
- Remember to `wandb login` first
- `./examples/data_preprocess/kk.py` prompt still need to be improved



