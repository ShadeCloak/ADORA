
# ADORA_VL  

## Quick Start 

### Docker Run
~~~bash  
docker pull docker.io/luc4gui/adora-mm-openrlhf:0.6.0.ds
docker run --gpus all -it --name adora-container docker.io/luc4gui/adora-mm-openrlhf:0.6.0.ds  
~~~

### Data Preparation

~~~bash  
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
./hfd.sh hiyouga/geometry3k
~~~

### Training

~~~bash  
git clone https://github.com/ShadeCloak/ADORA.git
cd ADORA
bash ADORA_VL/examples/scripts/qwen25vl_7b_geo3k_adora.sh
~~~

### Evaluation

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for evaluation, and simply need to write the following configuration into the config.py file.

~~~python  
    "Qwen2.5-VL-7B-Geo3k-ADORA-v0": partial(
        Qwen2VLChat,
        model_path="AdoraRL/Qwen2.5-VL-7B-Geo3k-ADORA-v0",
        min_pixels=3136,
        max_pixels=12845056,
        system_prompt="Please reason step by step, and put your final answer within \boxed{}.",
    ),
~~~

The evaluation results for MathVista are displayed in `ADORA/ADORA_VL/docs/math_results.xlsx`.

##  Weight Function
In the blog, weight_func is defined as follows, feel free to modify it according to your own scenario.
~~~python  
def weight_func(rewards, response_length, lamda=0.1):  
    """   
    """  
    weights = torch.ones_like(rewards, device=rewards.device)  
    for i in range(rewards.shape[0]):  
        reward_row = rewards[i]  
        response_row = response_length[i]  
        response_reward_1 = response_row[reward_row > 0.5]  
        response_reward_not_1 = response_row[(reward_row <= 0.5) & (response_row < 4094)]  
        max_reward_1 = response_reward_1.max() if response_reward_1.numel() > 0 else float('-inf')  
        mean_reward_not_1 = response_reward_not_1.mean() if response_reward_not_1.numel() > 0 else float('inf') 

        if max_reward_1 > mean_reward_not_1 or response_reward_1.numel() == 0:  
            pass
        else: 
            weights[i] = lamda

    return weights
~~~
