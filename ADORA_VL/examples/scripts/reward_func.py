import re  
import torch  

def reward_func(queries, prompts, ref):  
    """  
    """  
    assistant_queries = [re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", query, re.DOTALL)[-1] for query in queries]
    pattern = r'.*<think>.+?</think>\s*\S+'  
    rewards = [1 if re.match(pattern, query, re.DOTALL) else 0 for query in assistant_queries]  

    return torch.tensor(rewards, dtype=torch.float)