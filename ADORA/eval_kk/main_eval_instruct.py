      
import argparse
import json
import os
import numpy as np
import random
import torch
import time
from vllm import LLM, SamplingParams
from kk_processor import KKProcessor
import datasets
from concurrent.futures import ThreadPoolExecutor

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

def write_jsonl(output_file, data):
    """Write data to a JSONL file."""
    with open(output_file, "w+") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")

def init_seed(seed=42):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_eval_records(args, subject):
    """Load evaluation records based on arguments."""
    if args.problem_type != "clean":
        records = datasets.load_dataset('K-and-K/perturbed-knights-and-knaves', 
                                        data_files=f"{args.split}/{args.problem_type}/{subject}.jsonl")["train"]
    else:
        records = datasets.load_dataset('K-and-K/knights-and-knaves', 
                                        data_files=f"{args.split}/{subject}.jsonl")["train"]
    return records

def eval_subject(args, subject, llm, test_records, kk_proc, exist_result_records):
    """Evaluate one subject."""
    cors = []
    start_index = len(exist_result_records)
    print(f"Processing {subject} starting from index {start_index}")

    # Prepare all prompts using multi-threading
    prompts, labels = [], []
    new_prompts = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(kk_proc.gen_test_prompt, args.ntrain, test_records, i, args.model)
            for i in range(start_index, len(test_records))
        ]
        for future in futures:
            prompt, label = future.result()
            prompts.append(prompt)
            labels.append(label)
            question = prompt.split("### Question:")[1].strip().split("### Answer:")[0].strip()
            new_prompt = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) ...\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
            new_prompts.append(new_prompt)
    # Batch inference using vLLM
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_token,
    )
    outputs = llm.generate(new_prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    # Process results and save to exist_result_records
    for i, (new_prompt, label, response) in enumerate(zip(new_prompts, labels, responses), start=start_index):
        cor, parsed_pred, reformat_gold_conditions = kk_proc._parse_cot_eval_instruct(response, label, args.model, test_records[i]['names'], test_records[i]['solution_text_format'])
        cors.append(cor)
        new_item = {
            'quiz': test_records[i]['quiz'],
            'names': test_records[i]['names'],
            'solution': test_records[i]['solution'],
            'solution_text': test_records[i]['solution_text'],
            'solution_text_format': test_records[i]['solution_text_format'],
            'index': test_records[i]['index'],
            'predicts': parsed_pred,
            'labels': reformat_gold_conditions,
            'correct': cor,
            'response': response,
            'prompts': new_prompt,
        }
        exist_result_records.append(new_item)

    acc = np.mean(cors)
    print(f"Subject: {subject}, Accuracy: {acc:.3f}")
    return cors, acc, exist_result_records

def load_limited_test_records(args, subject, exist_result_records):
    """Load limited test records based on given arguments."""
    test_records = load_eval_records(args, subject)
    if args.limit is not None:
        test_records = test_records.select(range(min(args.limit, len(test_records))))
        if args.limit <= len(exist_result_records):
            return None  # Already processed all samples
    return test_records

def save_final_acc_results(all_cors, results, fname):
    """Save final accuracy results to a file."""
    if all_cors:
        weighted_acc = np.mean(np.concatenate(all_cors))
        results["weighted_accuracy"] = weighted_acc
        print(f"Overall Average Accuracy: {weighted_acc:.3f}")
        with open(fname, "w") as f:
            json.dump(results, f)

def load_previous_acc_results(fname):
    """Load previous accuracy results."""
    acc_results = {"subject": {}}
    if os.path.isfile(fname):
        with open(fname, 'r', encoding='utf-8') as file:
            acc_results = json.load(file)
    return acc_results

def get_subjects_to_eval(args):
    """Get subjects to evaluate."""
    subjects = []
    if args.split == "test":
        if args.eval_nppl == 0:
            subjects = [f"people{nppl}_num100" for nppl in range(2, 9)]
        else:
            subjects = [f"people{args.eval_nppl}_num100"]
    elif args.split == "train":
        if args.eval_nppl == 2:
            subjects = ["people2_num200"]
        elif args.eval_nppl > 2:
            subjects = [f"people{args.eval_nppl}_num1000"]
    return subjects

def main(args):
    model_short_name = "/".join(args.model.split("/")[-2:])
    prefix = os.path.join(args.save_dir, f"{model_short_name}_{args.ntrain}shot")
    args.config += f"_token{args.max_token}{('_cot' if args.cot else '')}" \
                   f"_{args.split}{('_' + args.problem_type if args.problem_type != 'clean' else '')}"
    output_folder = os.path.join(prefix, args.config)
    acc_fname = os.path.join(prefix, f"result_{args.config}.json")
    os.makedirs(output_folder, exist_ok=True)

    print("Configuration:", args.config)
    print("Output Folder:", output_folder)

    kk_proc = KKProcessor(cot=args.cot, no_linebreak=args.no_linebreak)
    subjects = get_subjects_to_eval(args)
    acc_results = load_previous_acc_results(acc_fname)

    # Initialize vLLM model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.ngpus,
        max_model_len=args.max_token,
        gpu_memory_utilization=0.85,
        enforce_eager=False
    )

    all_cors = []
    for subject in subjects:
        result_outfile = os.path.join(output_folder, f"{subject}.jsonl")
        exist_result_records = load_jsonl(result_outfile)
        test_records = load_limited_test_records(args, subject, exist_result_records)
        if test_records is None:
            continue

        cors, acc, result_records = eval_subject(args, subject, llm, test_records, kk_proc, exist_result_records)
        write_jsonl(result_outfile, result_records)
        all_cors.append(cors)
        acc_results["subject"][subject] = acc

    save_final_acc_results(all_cors, acc_results, acc_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for KK dataset")
    parser.add_argument("--ntrain", "-k", type=int, default=0, help="Number of training examples")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", "-s", type=str, default="result_qa", help="Save directory")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name or path")
    parser.add_argument("--max_token", type=int, default=1024, help="Maximum number of tokens")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--no_linebreak", action="store_true", help="Remove line breaks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for VLLM")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Data split to use")
    parser.add_argument("--eval_nppl", type=int, default=0, help="Number of people to evaluate")
    parser.add_argument("--problem_type", type=str, default="clean", help="Problem perturbation type")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--config", type=str, default="default_config", help="Configuration string for saving results")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs for parallel inference")
    args = parser.parse_args()

    init_seed()
    main(args)

    