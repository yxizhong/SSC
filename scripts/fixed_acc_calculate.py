#!/usr/bin/env python3
"""
Optimized TopK accuracy calculation with intermediate result reuse.
Only requires: res_dir, pattern, voting_methods, save_path parameters.
"""

import os
import sys
import glob
import pickle
import json
import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool, set_start_method
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer

# Add path for model.utils
sys.path.insert(0, "/share/yangxizhong/workspace/step_level_deepconf")
from model.utils import (
    calculate_mean_confidence,
    calculate_tail_confidence,
    calculate_bottom_window_confidence,
    filter_top_confidence,
    equal_func,
    compute_all_voting_results,
    compute_least_step,
    compute_least_grouped
)
from dynasor.core.evaluator import math_equal

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

def load_data_parallel(file_paths: List[str], num_workers: int = 4) -> Dict[int, Dict]:
    """Load pickle data in parallel."""
    def load_single_file(file_path: str) -> Tuple[int, Dict]:
        try:
            # Try torch.load first for CUDA tensors, fallback to pickle
            try:
                data = torch.load(file_path, map_location='cpu')
            except:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            return data["qid"], data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    pkl_data = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(load_single_file, file_paths),
            total=len(file_paths),
            desc="Loading files"
        ))

    for qid, data in results:
        if qid is not None and qid not in pkl_data:
            pkl_data[qid] = data
        else:
            pkl_data[qid]["warmup_traces"].extend(data["warmup_traces"])
            pkl_data[qid]["final_traces"].extend(data["final_traces"])
            pkl_data[qid]["all_traces"].extend(data["all_traces"])

    return pkl_data

class GPUVotingCalculator:
    """GPU-accelerated voting calculator using PyTorch tensors."""

    def __init__(self, traces: List[Dict[str, Any]], device: str = 'cuda', interval=False, mode="step"):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.traces, self.wramup_min_confs = traces
        self.mode = mode
        # self.valid_traces = [trace for trace in self.traces if trace.get('extracted_answer')]
        if self.wramup_min_confs and interval:
            self.valid_traces = self.traces[:len(self.wramup_min_confs)-1]
            conf_bar = float(np.percentile(self.wramup_min_confs[:-1], 100 - 10))
            final_trace = [trace for trace in self.traces if "conf_bar" in trace and trace["conf_bar"]==conf_bar]
            self.valid_traces.extend(final_trace)
        else:
            self.valid_traces = self.traces

        model_path = "/share/yangxizhong/ckpt/DeepSeek-R1-0528-Qwen3-8B"
        self.step_tokens = ["\n\n", ]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if not self.valid_traces:
            self.answers = []
            self.confidence_tensor = None
            return

        self.answers = [trace['extracted_answer'] for trace in self.valid_traces]

        # Precompute all confidences and move to GPU
        self._precompute_confidences_gpu()

    def _precompute_confidences_gpu(self):
        """Precompute all confidence types on GPU for fast access."""
        n_traces = len(self.valid_traces)

        # Initialize confidence matrix [n_traces, n_confidence_types]
        # Types: mean, tail, bottom_window, min_window
        confidences = torch.zeros((n_traces, 4), device=self.device, dtype=torch.float32)

        for i, trace in enumerate(self.valid_traces):
            if "group_confs" not in trace.keys() and trace["extracted_answer"]:
                if self.mode == "step":
                    trace["group_confs"] = compute_least_step(trace["token_ids"], trace["confs"], [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens])
                elif self.mode == "window":
                    trace["group_confs"] = compute_least_grouped(trace["confs"], group_size=2048)

            confidences[i, 0] = calculate_mean_confidence(trace) # mean
            confidences[i, 1] = calculate_tail_confidence(trace) # tail
            confidences[i, 2] = calculate_bottom_window_confidence(trace) # bottom
            confidences[i, 3] = calculate_bottom_window_confidence(trace, bottom_percent=-1) # lowest

        self.confidence_tensor = confidences

        # sorted_indices = torch.argsort(self.confidence_tensor[:,3])
        # self.confidence_tensor = self.confidence_tensor[sorted_indices, :]
        # self.valid_traces = [self.valid_traces[idx] for idx in sorted_indices]
        # self.answers = [self.answers[idx] for idx in sorted_indices]

        # Precompute filtered indices for top10 methods
        self._precompute_filtered_indices()

    def _precompute_filtered_indices(self):
        """Precompute filtered trace indices for top10 methods."""
        if self.confidence_tensor is None:
            self.top_tail_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.top_bottom_indices = torch.tensor([], dtype=torch.long, device=self.device)
            return

        n_traces = len(self.valid_traces)

        # Top 10% tail confidence filtered
        tail_confs = self.confidence_tensor[:, 1]  # tail confidence column
        if len(tail_confs) > 0:
            top_tail_threshold = torch.quantile(tail_confs, 0.9)
            self.top_tail_indices = torch.where(tail_confs >= top_tail_threshold)[0]
        else:
            self.top_tail_indices = torch.tensor([], dtype=torch.long, device=self.device)

        # Top 10% bottom window confidence filtered
        bottom_confs = self.confidence_tensor[:, 2]  # bottom window confidence column
        if len(bottom_confs) > 0:
            top_bottom_threshold = torch.quantile(bottom_confs, 0.9)
            self.top_bottom_indices = torch.where(bottom_confs >= top_bottom_threshold)[0]
        else:
            self.top_bottom_indices = torch.tensor([], dtype=torch.long, device=self.device)

    def weighted_majority_vote_gpu(self, answers: List[str], weights: torch.Tensor) -> Optional[str]:
        """GPU-accelerated weighted majority voting."""
        if not answers or weights.numel() == 0:
            return None

        # Convert to CPU for string operations
        weights_cpu = weights.cpu().numpy()

        answer_weights = {}
        for answer, weight in zip(answers, weights_cpu):
            if answer is not None:
                answer_str = str(answer)
                answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)

        if not answer_weights:
            return None

        return max(answer_weights.keys(), key=lambda x: answer_weights[x])
    
    def warmup_with_k_indices(self, k, sorted_indices=None, interval=None):
        if sorted_indices is not None and interval:
            topk_trace_indices = sorted_indices[interval[0]:interval[1]]
        elif self.wramup_min_confs is None:
            topk_trace_indices = list(range(k))
        else:
            k -= 1
            warmup_traces_indices = [idx for idx, trace in enumerate(self.traces) if "conf_bar" not in trace]
            final_traces_indices = [idx for idx, trace in enumerate(self.traces) if "conf_bar" in trace]
            if k==0:
                topk_trace_indices = [0]
            else:
                warmup_traces_min_confs = float(np.percentile(self.wramup_min_confs[:k], 100 - 10))
                topk_trace_indices = list(range(k)) + [idx for idx in final_traces_indices if self.traces[idx]["conf_bar"]==warmup_traces_min_confs]
        
        # print(f"ori K={len(topk_trace_indices)}, {topk_trace_indices}")
        topk_trace_indices = [idx for idx in topk_trace_indices if self.answers[idx] is not None]
        # print(f"voting K={len(topk_trace_indices)}, {topk_trace_indices}")
        return topk_trace_indices
    
    def compute_top_threshold_by2derivative(self, confs, window_size=30):
        if confs.is_cuda:
            confs = confs.cpu()
        else:
            confs = confs

        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        sorted_values, _ = torch.sort(confs, descending=False)
        sorted_values_np = sorted_values.numpy()
        first_derivative = np.diff(sorted_values_np)

        if len(first_derivative) >= window_size:
            smoothed_derivative = moving_average(first_derivative, window_size)
        else:
            smoothed_derivative = first_derivative
        
        second_derivative = np.diff(smoothed_derivative)
        if len(second_derivative) >= window_size:
            smoothed_second_derivative = moving_average(second_derivative, window_size)
        else:
            smoothed_second_derivative = second_derivative
        
        record = [idx for idx,_ in enumerate(smoothed_second_derivative) if _ < 0]
        if len(record):
            top_idx = record[-1]
            top_threshold = confs[top_idx]
        else:
            top_idx = 1
            top_threshold = 0            

        return top_threshold, top_idx
        

    def compute_top_threshold_adaptive(self, confs):
        if isinstance(confs, torch.Tensor):
            values = confs.cpu().numpy()
        else:
            values = confs
        sorted_order = torch.argsort(confs, descending=False)
        sorted_values = confs[sorted_order]
        
        if len(sorted_values) <= 1:
            return (sorted_values[0].item(),1) if len(sorted_values) == 1 else (0,0)
        
        max_change = 0
        max_change_conf = sorted_values[0].item()
        max_idx = 0

        for i in range(len(sorted_values) - 1):
            current_val = sorted_values[i].item()
            next_val = sorted_values[i + 1].item()
            
            if current_val != 0:
                relative_change = abs((next_val - current_val))
            else:
                relative_change = float('inf') if next_val != 0 else 0
            
            if relative_change > max_change:
                max_change = relative_change
                max_change_conf = next_val
                max_idx = i+2
        
        return max_change_conf, max_idx
        
    def compute_voting_results_topk_gpu(self, k: int, voting_methods: List[str], sorted_indices=None, interval=None) -> Dict[str, Any]:
        """GPU-accelerated voting results computation for top-k traces."""
        if k == 0 or not self.valid_traces or self.confidence_tensor is None:
            return {method: None for method in voting_methods}

        k = min(k, len(self.valid_traces))

        if k == len(self.valid_traces):
            print()

        # Get top-k data
        topk_answers = self.answers[:k]
        topk_confidences = self.confidence_tensor[:k]  # [k, 4]
        topk_indices = self.warmup_with_k_indices(k, sorted_indices, interval)
        topk_answers = [self.answers[idx] for idx in topk_indices]
        topk_confidences = self.confidence_tensor[topk_indices,:]
        
        self.top_type_indices = []
        adaptive_topk_idxs = []
        for traj_conf_idx in range(self.confidence_tensor.size()[-1]):
            type_confs = self.confidence_tensor[topk_indices, traj_conf_idx]
            if len(type_confs) > 0:
                # top_type_threshold, adaptive_topk_idx = self.compute_top_threshold_adaptive(type_confs)
                # top_type_threshold, adaptive_topk_idx = self.compute_top_threshold_by2derivative(type_confs)
                top_type_threshold = torch.quantile(type_confs, 0.9)
                adaptive_topk_idx = [idx for idx, res in enumerate(type_confs >= top_type_threshold) if res][0]
                top_type_indices = torch.tensor([topk_indices[idx] for idx, res in enumerate(type_confs >= top_type_threshold) if res])
                adaptive_topk_idxs.append(adaptive_topk_idx)
            else:
                adaptive_topk_idxs.append(0)
                top_type_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.top_type_indices.append(top_type_indices)

        voting_results = {}

        for method in voting_methods:
            if method == 'majority':
                vote_counts = Counter(topk_answers)
                voting_results[method] = {
                    'answer': vote_counts.most_common(1)[0][0] if vote_counts else None,
                    'num_votes': len(topk_answers),
                    'confidence': None
                }

            elif method == 'mean_confidence_weighted':
                weights = topk_confidences[:, 0]  # mean confidence
                if torch.any(weights > 0):
                    voting_results[method] = {
                        'answer': self.weighted_majority_vote_gpu(topk_answers, weights),
                        'num_votes': len(topk_answers),
                        'confidence': float(torch.mean(weights))
                    }
                else:
                    voting_results[method] = {}

            elif method == 'tail_confidence_weighted':
                weights = topk_confidences[:, 1]  # tail confidence
                if torch.any(weights > 0):
                    voting_results[method] = {
                        'answer': self.weighted_majority_vote_gpu(topk_answers, weights),
                        'num_votes': len(topk_answers),
                        'confidence': float(torch.mean(weights))
                    }
                else:
                    voting_results[method] = {}

            elif method == 'bottom_window_weighted':
                weights = topk_confidences[:, 2]  # bottom window confidence
                if torch.any(weights > 0):
                    voting_results[method] = {
                        'answer': self.weighted_majority_vote_gpu(topk_answers, weights),
                        'num_votes': len(topk_answers),
                        'confidence': float(torch.mean(weights))
                    }
                else:
                    voting_results[method] = {}

            elif method == 'min_window_weighted':
                weights = topk_confidences[:, 3]  # min window confidence
                if torch.any(weights > 0):
                    voting_results[method] = {
                        'answer': self.weighted_majority_vote_gpu(topk_answers, weights),
                        'num_votes': len(topk_answers),
                        'confidence': float(torch.mean(weights))
                    }
                else:
                    voting_results[method] = {}

            elif method == 'top10_tail_filtered':
                # Apply k limit to filtered indices
                # filtered_indices = self.top_tail_indices[self.top_tail_indices < k]
                filtered_indices = self.top_type_indices[1]
                if len(filtered_indices) > 0:
                    filtered_answers = [self.answers[i] for i in filtered_indices.cpu()]
                    filtered_weights = self.confidence_tensor[filtered_indices, 1]  # tail confidence
                    topk_idx = adaptive_topk_idxs[1]
                    if torch.any(filtered_weights > 0):
                        voting_results[method] = {
                            'answer': self.weighted_majority_vote_gpu(filtered_answers, filtered_weights),
                            'num_votes': len(filtered_answers),
                            'confidence': float(torch.mean(filtered_weights)),
                            'topk_idx': topk_idx
                        }
                    else:
                        voting_results[method] = {}
                else:
                    voting_results[method] = {}

            elif method == 'top10_bottom_window_filtered':
                # Apply k limit to filtered indices
                # filtered_indices = self.top_bottom_indices[self.top_bottom_indices < k]
                filtered_indices = self.top_type_indices[2]
                topk_idx = adaptive_topk_idxs[2]
                if len(filtered_indices) > 0:
                    filtered_answers = [self.answers[i] for i in filtered_indices.cpu()]
                    filtered_weights = self.confidence_tensor[filtered_indices, 2]  # bottom window confidence

                    if torch.any(filtered_weights > 0):
                        voting_results[method] = {
                            'answer': self.weighted_majority_vote_gpu(filtered_answers, filtered_weights),
                            'num_votes': len(filtered_answers),
                            'confidence': float(torch.mean(filtered_weights)),
                            'topk_idx': topk_idx
                        }
                    else:
                        voting_results[method] = {}
                else:
                    voting_results[method] = {}
            
            elif method == 'top10_lowest_window_filtered':
                # Apply k limit to filtered indices
                # filtered_indices = self.top_bottom_indices[self.top_bottom_indices < k]
                filtered_indices = self.top_type_indices[3]
                if len(filtered_indices) > 0:
                    filtered_answers = [self.answers[i] for i in filtered_indices.cpu()]
                    filtered_weights = self.confidence_tensor[filtered_indices, 3]  # bottom window confidence
                    topk_idx = adaptive_topk_idxs[3]
                    if torch.any(filtered_weights > 0):
                        voting_results[method] = {
                            'answer': self.weighted_majority_vote_gpu(filtered_answers, filtered_weights),
                            'num_votes': len(filtered_answers),
                            'confidence': float(torch.mean(filtered_weights)),
                            'topk_idx': topk_idx
                        }
                    else:
                        voting_results[method] = {}
                else:
                    voting_results[method] = {}
            
            elif method == 'top10_mean_filtered':
                # Apply k limit to filtered indices
                # filtered_indices = self.top_bottom_indices[self.top_bottom_indices < k]
                filtered_indices = self.top_type_indices[0]
                if len(filtered_indices) > 0:
                    filtered_answers = [self.answers[i] for i in filtered_indices.cpu()]
                    filtered_weights = self.confidence_tensor[filtered_indices, 0]  # bottom window confidence
                    topk_idx = adaptive_topk_idxs[0]
                    if torch.any(filtered_weights > 0):
                        voting_results[method] = {
                            'answer': self.weighted_majority_vote_gpu(filtered_answers, filtered_weights),
                            'num_votes': len(filtered_answers),
                            'confidence': float(torch.mean(filtered_weights)),
                            'topk_idx': topk_idx
                        }
                    else:
                        voting_results[method] = {}
                else:
                    voting_results[method] = {}

            else:
                voting_results[method] = {}

        return voting_results

def process_single_question_gpu(args: Tuple[int, List[Dict], str, List[str]], interval=False, mode="step") -> Tuple[int, Dict]:
    """Process a single question with GPU acceleration."""
    qid, traces, gt, voting_methods = args

    if not traces:
        return qid, {method: [] for method in voting_methods}

    # Create GPU calculator
    calculator = GPUVotingCalculator(traces, device='cuda' if torch.cuda.is_available() else 'cpu', interval=interval, mode=mode)

    # Initialize results
    question_results = {method: [] for method in voting_methods}

    if calculator.wramup_min_confs is not None:
        max_k = max(512,len(calculator.wramup_min_confs))
    else:
        max_k = len(calculator.traces)

    if max_k == 0:
        return qid, question_results
    
    if interval:
        question_results = {method: {} for method in voting_methods}
        n=len(calculator.confidence_tensor[:,0])
        intervals = [(round(i * n * 0.05), round((i + 1) * n * 0.05) - 1) for i in range(int(1 / 0.05))]
        sorted_methods_map = {
            0: "mean",
            1: "tail",
            2: "bottom",
            3: "lowest",
        }
        for i in range(4):
            sorted_indices = torch.argsort(calculator.confidence_tensor[:,i])

            for interval in intervals:
                voting_results = calculator.compute_voting_results_topk_gpu(512, voting_methods, sorted_indices, interval)

                for method in voting_methods:
                    if voting_results[method] is not None:
                        answer = voting_results[method]["answer"]
                        if answer is not None:
                            correct = equal_func(answer, gt)
                        else:
                            correct = False
                    else:
                        correct = False
                    
                    if sorted_methods_map[i] not in question_results[method]:
                        question_results[method][sorted_methods_map[i]]=[correct]
                    else:
                        question_results[method][sorted_methods_map[i]].append(correct)
    else:
        # Calculate accuracy for each k from 1 to max_k
        for k in range(1, max_k + 1):
            voting_results = calculator.compute_voting_results_topk_gpu(k, voting_methods)

            for method in voting_methods:
                if voting_results[method] is not None:
                    answer = voting_results[method].get("answer", None)
                    if answer is not None:
                        correct = equal_func(answer, gt)
                    else:
                        correct = False
                else:
                    correct = False
                voting_results[method]["correctness"] = correct

                question_results[method].append(voting_results[method])

    return qid, question_results

def _calculate_topk_accuracy(res_dir: str, pattern: str, voting_methods: List[str],
                           save_path: str, num_workers: int = None, use_gpu: bool = True, mode="step"):
    """
    Optimized TopK accuracy calculation with GPU acceleration and parallelization.

    Args:
        res_dir: Directory containing pickle files
        pattern: File pattern to match
        voting_methods: List of voting methods to evaluate
        save_path: Path to save results
        num_workers: Number of parallel workers (None for auto)
        use_gpu: Whether to use GPU acceleration
    """

    # Set up multiprocessing
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Limit to avoid memory issues

    print(f"Using {num_workers} workers, GPU: {use_gpu and torch.cuda.is_available()}")

    # Load data with parallel loading
    pattern_path = os.path.join(res_dir, pattern)
    file_paths = natsorted(glob.glob(pattern_path))
    # tmp = {}
    # for path in file_paths:
    #     if path[:-20] not in tmp:
    #         tmp[path[:-20]] = path
    # file_paths = list(tmp.values())
    print(f"Loading data from {len(file_paths)} files...")
    pkl_data = load_data_parallel(file_paths, num_workers=num_workers)

    # Extract voting traces and ground truth
    voting_traces = {}
    ground_truth = {}

    for trace_data in pkl_data.values():
        qid = trace_data["qid"]
        traces = []

        for trace in trace_data["all_traces"]:
        # for trace in trace_data["warmup_traces"]:
            traces.append(trace)
        
        if len(trace_data["final_traces"]) != 1 and len(trace_data["warmup_traces"]) > 0:
            voting_traces[qid] = [traces, trace_data["warmup_min_confs"]]
        else:
            voting_traces[qid] = [traces, None]
        ground_truth[qid] = trace_data["ground_truth"]

    print(f"Loaded {len(voting_traces)} questions")

    # Filter out already processed questions
    questions_to_process = []
    for qid in ground_truth.keys():
        if should_skip_qid(qid, save_path):
            print(f"Skipping question {qid}")
            continue

        if qid in voting_traces:
            questions_to_process.append((qid, voting_traces[qid], ground_truth[qid], voting_methods))

    print(f"Processing {len(questions_to_process)} questions")

    if use_gpu and torch.cuda.is_available():
        # GPU processing - process questions sequentially on GPU
        print("Using GPU acceleration")
        for args in tqdm(questions_to_process, desc="Processing questions on GPU"):
            qid, question_results = process_single_question_gpu(args, interval=False, mode=mode)

            res = {
                "qid": qid,
                "evaluation_res": question_results,
            }
            save_detail_res2json(res, save_path)

    else:
        # CPU parallel processing
        print(f"Using CPU parallel processing with {num_workers} workers")

        # Process in chunks to avoid memory issues
        chunk_size = max(1, len(questions_to_process) // (num_workers * 2))

        for i in tqdm(range(0, len(questions_to_process), chunk_size), desc="Processing chunks"):
            chunk = questions_to_process[i:i + chunk_size]

            with ProcessPoolExecutor(max_workers=min(num_workers, len(chunk))) as executor:
                results = list(executor.map(process_single_question_gpu, chunk))

            # Save results immediately
            for qid, question_results in results:
                res = {
                    "qid": qid,
                    "evaluation_res": question_results,
                }
                save_detail_res2json(res, save_path)
 

if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    res_dir = "/share/yangxizhong/output/deepconf/stepconf_deepthink/gpqad-low-warmupK-parallel-20251202"
    pattern = "deepthink_online*.pkl"
    voting_methods = [
        "majority",
        "mean_confidence_weighted",
        "tail_confidence_weighted",
        "bottom_window_weighted",
        "min_window_weighted",
        "top10_tail_filtered",
        "top10_bottom_window_filtered",
        "top10_lowest_window_filtered",
        "top10_mean_filtered"
    ]
    save_path = os.path.join(res_dir, "fixed_acc_test_H800_stepbased_Alltop10_20251204.json")

    # Use optimized version with GPU acceleration and parallel processing
    _calculate_topk_accuracy(
        res_dir=res_dir,
        pattern=pattern,
        voting_methods=voting_methods,
        save_path=save_path,
        num_workers=8,  # Adjust based on your system
        use_gpu=True,   # Set to False to use CPU only
        mode="step"
    )