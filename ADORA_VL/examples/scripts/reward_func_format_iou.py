import re  
import torch  

def reward_func(queries, prompts, solutions):  
    """  
    iou_reward & format_reward in: 
    https://github.com/om-ai-lab/VLM-R1/blob/main/src/open-r1-multimodal/src/open_r1/grpo_rec.py
    """  

    assistant_queries = [re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", query, re.DOTALL)[-1] for query in queries]

    pattern_format = r".*?<think>.*?</think>.*?<answer>.*?</answer>.*?"  
    rewards_format = [1 if re.match(pattern_format, query, re.DOTALL) else 0 for query in assistant_queries]  

    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    rewards = []
    for content, sol in zip(assistant_queries, solutions):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                sol_match = re.search(bbox_pattern, sol)
                if bbox_match:
                    bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
                    sol_bbox = [float(sol_match.group(1)), float(sol_match.group(2)), float(sol_match.group(3)), float(sol_match.group(4))]
                    if iou(bbox, sol_bbox) > 0.5:
                        reward = 1.0
        except Exception as e:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)

    return torch.tensor(rewards_format, dtype=torch.float) + torch.tensor(rewards, dtype=torch.float)


if __name__ == "__main__":
    # test utils
    queries = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|vision_end|>Please provide the bounding box coordinate of the region this sentence describes: the pictures didnt matchthis one is number 10. First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.<|im_end|>\n<|im_start|>assistant\n```json\n{\n    \"thinking_process\": \"<think> There is a soccer team scene captured during a nighttime game. The player in the white jacket with the number 10 on the back is at the center of action. The athlete in white is running towards the ball, has just kicked it, as the defender from the red/red socks team is attempting to block him. All players are focused on the ball in motion. The medium-sized ball is positioned on the field near the center. </think>\",\n    \"answer\": \"<answer> [364, 66, 603, 350] </answer>\"\n}\n```<|im_end|>"
    prompts = ""
    solutions = "[385.53,60.03,600.5,357.19]"
    score = reward_func([queries], [prompts], [solutions])
    print(score)