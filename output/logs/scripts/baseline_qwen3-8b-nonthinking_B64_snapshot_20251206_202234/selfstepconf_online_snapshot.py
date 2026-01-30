"""
Online baseline for DeepThinkLLM - processes a single question without early exit
Based on offline mode but structured for online-style evaluation

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import json
import sys
sys.path.insert(0, "/share/yangxizhong/workspace/step_level_deepconf")
import pickle
import argparse
from datetime import datetime
from vllm import SamplingParams
from model.wrapper import DeepThinkLLM
from dynasor.core.evaluator import math_equal
import time
from tqdm import tqdm

# ============= PROMPT PREPARATION FUNCTIONS =============

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek", datasets: str = "hmmt") -> str:
    """Prepare prompt for a single question"""
    if model_type == "deepseek":
        # Format prompt using chat template for DeepSeek
        if "gpqa" in datasets:
            messages = [
                {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
                {"role": "user", "content": f"Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response. {question}"}
                ]
        else:
            messages = [
                {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
                {"role": "user", "content": question}
                ]
    elif model_type == "qwen":
        if "gpqa" in datasets:
            messages = [    
                {"role": "user", "content": f"Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response. {question}"}
            ]
        else:
            messages = [
                {"role": "user", "content": question + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
            ]
    else:
        # Format for GPT-like models
        messages = [
            {"role": "user", "content": question}
        ]
    print(messages)
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=False
    )
    
    return full_prompt

def prepare_prompt_gpt(question: str, tokenizer, reasoning_effort: str = "high") -> str:
    """Prepare prompt for GPT models with reasoning effort"""
    messages = [
        {"role": "user", "content": question}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
    )
    
    return full_prompt


def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def evaluate_voting_results(voting_results, ground_truth):
    """Evaluate voting results against ground truth"""
    evaluation = {}
    
    for method, result in voting_results.items():
        # if "top10" in method:
        #     continue
        if result and result.get('answer'):
            try:
                is_correct = equal_func(result['answer'], ground_truth)
            except:
                is_correct = str(result['answer']) == str(ground_truth)
            
            evaluation[method] = {
                'answer': result['answer'],
                'is_correct': is_correct,
                'confidence': result.get('confidence'),
                'num_votes': result.get('num_votes', 0)
            }
        else:
            evaluation[method] = {
                'answer': None,
                'is_correct': False,
                'confidence': None,
                'num_votes': 0
            }
    
    return evaluation

def save_detail_res2json(res, save_path=None):
    if save_path is None:
        return None
    if not os.path.exists(save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([res], f, ensure_ascii=False, indent=4)
    else:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.append(res)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print("save question success: ", res["qid"])

def print_evaluation_report(question, ground_truth, evaluation, result, qid, detail_res_json_file=None):
    """Print detailed evaluation report"""
    print(f"\n=== Online Baseline Evaluation Report ===")
    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"Total traces generated: {result.total_traces_count}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Generation time: {result.generation_time:.2f}s")
    
    if result.generation_time > 0:
        print(f"Generation throughput: {result.total_tokens / result.generation_time:.1f} tokens/second")
    
    # Overall individual trace accuracy
    correct_traces = sum(1 for trace in result.all_traces 
                        if trace.get('extracted_answer') and 
                        equal_func(trace['extracted_answer'], ground_truth))
    total_valid_traces = sum(1 for trace in result.all_traces if trace.get('extracted_answer'))
    
    if total_valid_traces > 0:
        overall_accuracy = correct_traces / total_valid_traces
        print(f"Overall individual trace accuracy: {correct_traces}/{total_valid_traces} ({overall_accuracy:.1%})")
    
    
    # Voting method results
    print(f"\n=== Voting Method Results ===")
    print("-" * 80)
    print(f"{'Method':<25} {'Answer':<20} {'Correct':<8} {'Confidence':<12} {'Votes':<6}")
    print("-" * 80)
    
    correct_methods = []

    res = {
        "qid": qid,
        "question": question,
        "ground_truth": ground_truth,
    }

    evaluation_res = {}
    for method, eval_result in evaluation.items():
        answer = str(eval_result['answer'])[:18] + '...' if len(str(eval_result['answer'])) > 20 else str(eval_result['answer'])
        is_correct = eval_result['is_correct']
        confidence = eval_result['confidence']
        num_votes = eval_result['num_votes']
        
        correct_str = '✓' if is_correct else '✗'
        conf_str = f"{confidence:.3f}" if confidence is not None else '-'
        
        if answer:
            print(f"{method:<25} {answer:<20} {correct_str:<8} {conf_str:<12} {num_votes:<6}")
        else:
            print(f"{method:<25} {' ':<20} {correct_str:<8} {conf_str:<12} {num_votes:<6}")
        evaluation_res[method] = {
            "answer": answer,
            "correct": is_correct,
            "confidence": confidence,
            "votes": num_votes
        }
        if is_correct:
            correct_methods.append(method)
    res["evaluation_res"] = evaluation_res
    save_detail_res2json(res, detail_res_json_file)
    print(f"\nCorrect voting methods: {correct_methods}")
    
    # Find best method by confidence among correct ones
    correct_evals = {method: eval_result for method, eval_result in evaluation.items() 
                    if eval_result['is_correct']}
    
    if correct_evals:
        best_method = max(correct_evals.items(), 
                         key=lambda x: x[1]['confidence'] if x[1]['confidence'] is not None else 0)
        print(f"Best correct method: {best_method[0]} (confidence: {best_method[1]['confidence']:.3f})")
    
    # Method performance summary
    total_methods = len(evaluation)
    correct_count = len(correct_methods)
    print(f"Voting method accuracy: {correct_count}/{total_methods} ({correct_count/total_methods:.1%})")
    

def should_skip_qid(qid, json_file):
    if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
        return False

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return False
    except Exception as e:
        print(f"[Warning] Failed to read {json_file}: {e}")
        return False

    # 检查 qid 是否已存在
    for item in data:
        if isinstance(item, dict) and item.get("qid") == qid:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description='DeepThinkLLM Online Baseline (No Early Exit)')
    parser.add_argument('--model', type=str, default="/share/yangxizhong/ckpt/DeepSeek-R1-0528-Qwen3-8B",
                       help='Model path or name')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Tensor parallel size for model')
    parser.add_argument('--dataset', type=str, default="brumo_2025.jsonl",
                       help='Dataset file path')
    parser.add_argument('--qid', type=int, required=True,
                       help='Question ID to process (0-based index)')
    parser.add_argument('--rid', type=str, default="online_baseline",
                       help='Run ID for identification')
    parser.add_argument('--budget', type=int, default=256,
                       help='Total number of traces to generate (equivalent to total_budget in online mode)')
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Sliding window size for confidence computation')
    parser.add_argument('--max_tokens', type=int, default=64000,
                       help='Maximum tokens per generation')
    parser.add_argument('--model_type', type=str, default="deepseek", choices=["deepseek", "gpt"],
                       help='Model type for prompt formatting')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=0,
                       help='Top-k sampling parameter')
    parser.add_argument('--output_dir', type=str, default="outputs",
                       help='Output directory for results')
    parser.add_argument('--no_multiple_voting', action='store_true',
                       help='Disable multiple voting analysis')
    args = parser.parse_args()

    if "deepseek" in args.model.lower():
        args.model_type = "deepseek"
    elif "qwen" in args.model.lower():
        args.model_type = "qwen"

    datasets_name = args.output_dir.split('/')[-1].split('-')[0]
    detail_res_json_file = f"{args.output_dir}/{datasets_name}_dpsk_online.json"
    if should_skip_qid(args.qid, detail_res_json_file):
        print(f"skip: {args.qid}")
        return

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    if args.dataset.endswith("jsonl"):
        with open(args.dataset, 'r', encoding='utf-8') as file:
            data = [json.loads(line.strip()) for line in file]
    elif args.dataset.endswith("parquet"):
        import pandas as pd
        df = pd.read_parquet(args.dataset)
        if "problem" in df.columns:
            df = df.rename(columns={"problem": "question"})
        if "solution" in df.columns:
            df = df.rename(columns={"solution": "answer"})
        data = df.to_dict(orient="records")
    
    # Validate question ID
    if args.qid >= len(data) or args.qid < 0:
        raise ValueError(f"Question ID {args.qid} is out of range (0-{len(data)-1})")
    
    # Initialize DeepThinkLLM
    deep_llm = DeepThinkLLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True)

    # for idx in tqdm(range(len(data))):
    #     args.qid = idx
    question_data = data[args.qid]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    
    print(f"Processing question {args.qid}: {question[:100]}...")
    print(f"Budget: {args.budget} traces (no early exit)")

    # Prepare prompt
    print("Preparing prompt...")
    if args.model_type == "gpt":
        prompt = prepare_prompt_gpt(question, deep_llm.tokenizer, args.reasoning_effort)
    else:
        prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type, args.dataset)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20,
    )
    
    print(f"Running deep thinking baseline (offline mode, budget={args.budget})...")
    start_time = time.time()
    
    # Run deep thinking in offline mode (no early exit)
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",  # Use offline mode to avoid early exit
        budget=args.budget,
        window_size=args.window_size,
        sampling_params=sampling_params,
        compute_multiple_voting=not args.no_multiple_voting
    )
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f}s")
    
    # Evaluate results against ground truth
    evaluation = None
    
    if ground_truth:
        if result.voting_results:
            evaluation = evaluate_voting_results(result.voting_results, ground_truth)
        
        print_evaluation_report(question, ground_truth, evaluation, result, args.qid, detail_res_json_file)
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = result.to_dict()
    result_data.update({
        'question': question,
        'ground_truth': ground_truth,
        'qid': args.qid,
        'run_id': args.rid,
        'evaluation': evaluation,
        'mode': 'online_baseline_no_early_exit',
        'total_budget': args.budget
    })
    
    result_filename = f"{args.output_dir}/deepthink_online_baseline_qid{args.qid}_rid{args.rid}_{timestamp}.pkl"
    
    with open(result_filename, 'wb') as f:
        pickle.dump(result_data, f)
    
    print(f"\nResults saved to {result_filename}")
    
    # Print summary for easy comparison
    if ground_truth and evaluation:
        # Get best voting method accuracy
        correct_methods = [method for method, result in evaluation.items() if result['is_correct']]
        best_accuracy = "N/A"
        if correct_methods:
            best_accuracy = f"{len(correct_methods)}/{len(evaluation)} methods correct"
        
        print(f"\n=== BASELINE SUMMARY ===")
        print(f"Model: {args.model}")
        print(f"Question ID: {args.qid}")
        print(f"Total Budget: {args.budget}")
        print(f"Voting Methods: {best_accuracy}")
        print(f"Total Tokens: {result.total_tokens}")
        print(f"Generation Time: {result.generation_time:.2f}s")


if __name__ == "__main__":
    main()