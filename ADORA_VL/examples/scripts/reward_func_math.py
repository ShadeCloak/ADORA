import re
import torch
from mathruler.grader import extract_boxed_content, grade_answer


def math_compute_score(predict_str: str, ground_truth: str) -> float:
    """
    https://github.com/hiyouga/EasyR1/blob/main/verl/utils/reward_score/math.py#L4
    """
    answer = extract_boxed_content(predict_str)
    if answer == "None":
        return 0.0  # no answer

    if grade_answer(answer, ground_truth):
        return 1.0  # correct answer

    return 0.1  # wrong answer


def reward_func(queries, prompts, references):  
    """  
    """  
    assistant_queries = [re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", query, re.DOTALL)[-1] for query in queries]
    rewards = [math_compute_score(query, reference) for query, reference in zip(assistant_queries, references)]  

    return torch.tensor(rewards, dtype=torch.float)