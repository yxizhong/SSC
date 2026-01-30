
import re
import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dynasor.core.evaluator import math_equal

def simple_majority_vote(answers: List[str]) -> Optional[str]:
    """Simple majority voting"""
    if not answers:
        return None
    
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    # return mob_adaptive_m(answers, weights)

    """Perform weighted majority voting"""
    if not answers:
        return None
    
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])

def mob_adaptive_m(final_answers, 
                   rewards,
                   mob_q=0.75,
                   return_m_value = False,
                   **kwargs) -> int:
    def _bootstrap_bestofn_dist(final_answers, 
                            rewards, 
                            m: int, 
                            unique_final_answers=None):
        if unique_final_answers is None:
            unique_final_answers, counts = np.unique(final_answers, return_counts=True)
        dist = np.zeros(len(unique_final_answers))
        N = len(rewards)
        sorted_indices = np.argsort(rewards)
        sorted_indices_selctions_probs = ((np.arange(N) + 1) / N)**m - (np.arange(N) / N)**m
        for i in range(len(unique_final_answers)):
            final_ans = unique_final_answers[i]
            final_ans_sorted_indices = np.where((final_answers[sorted_indices] == final_ans))[0]
            dist[i] = np.sum(sorted_indices_selctions_probs[final_ans_sorted_indices])
        return dist, unique_final_answers
    final_answers = np.array(final_answers)
    n = len(final_answers)
    if n == 1:
        return 0
    unique_final_answers, counts = np.unique(final_answers, return_counts=True)
    m_candidates = set()
    for i in range(50):
        m = max(1, int((mob_q ** i) * n))
        m_candidates.add(m)
        if m == 1:
            break
    m_candidates = sorted(list(m_candidates))
    dists = np.zeros((len(m_candidates), len(unique_final_answers)))
    diffs = np.zeros((len(m_candidates) - 1,))
    min_diff = np.inf
    min_diff_i = None
    for i, m in enumerate(m_candidates):
        dist, _ = _bootstrap_bestofn_dist(final_answers, rewards, m, unique_final_answers)
        dists[i] = dist
        if i > 0:
            diffs[i - 1] = np.sum(np.abs(dists[i] - dists[i - 1]))
            if diffs[i - 1] < min_diff:
                min_diff = diffs[i - 1]
                min_diff_i = i

    dist = dists[min_diff_i]
    chosen_answer = unique_final_answers[np.argmax(dist)]
    if not return_m_value:
        return str(final_answers[np.where(final_answers == chosen_answer)[0][0]])
    else:
        return np.where(final_answers == chosen_answer)[0][0], m_candidates[min_diff_i]

def calculate_mean_confidence(trace: Dict[str, Any]) -> float:
    """Calculate mean confidence from confs in a trace"""
    try:
        if "group_confs" in trace and trace["group_confs"]:
            confs = trace["group_confs"]
            return np.mean(confs) if confs else 0.0
        elif 'confs' in trace and trace['confs']:
            confs = trace['confs']
            return np.mean(confs) if confs else 0.0
        elif "step_confs" in trace and trace["step_confs"]:
            confs = trace["step_confs"]
            return np.mean([np.mean(step_conf) for step_conf in confs]) if confs else 0.0

        return 0.0
    except Exception:
        return 0.0


def calculate_tail_confidence(trace: Dict[str, Any], tail_tokens: int = 2048) -> float:
    """Calculate mean confidence from the last N tokens"""
    try:
        if "group_confs" in trace and trace["group_confs"]:
            confs = trace["group_confs"]
            tail_confs = confs[-1]
            return np.mean(tail_confs) if tail_confs else 0.0
        elif 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        elif "step_confs" in trace and trace["step_confs"]:
            confs = trace["step_confs"]
            tail_confs = confs[-1]
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except Exception:
        return 0.0

def filter_top_confidence(traces: List[Dict[str, Any]], confidence_type: str = 'tail', top_percent: float = 0.1, mode="top") -> List[Dict[str, Any]]:
    """Filter traces by top confidence percentage"""
    if not traces:
        return []
    
    # Calculate confidences
    confidences = []
    for trace in traces:
        if confidence_type == 'mean':
            conf = calculate_mean_confidence(trace)
        elif confidence_type == 'tail':
            conf = calculate_tail_confidence(trace)
        elif confidence_type == 'bottom_window':
            conf = calculate_bottom_window_confidence(trace)
        elif confidence_type == 'min_window':
            conf = calculate_bottom_window_confidence(trace, bottom_percent=-1)
        else:
            conf = calculate_mean_confidence(trace)  # default fallback
        confidences.append(conf)
    
    # Get threshold for top percentage
    if ADAPTIVE_TYPE==0:
        threshold = np.percentile(confidences, (1 - top_percent) * 100)
    elif ADAPTIVE_TYPE==1:
        threshold,_ = compute_top_threshold_by1derivative(confidences)
    elif ADAPTIVE_TYPE==2:
        threshold,_ = compute_top_threshold_by2derivative(confidences)
    elif ADAPTIVE_TYPE==3:
        threshold = CONF_THRESHOLD
    # threshold = float(16)
    
    # Filter traces
    filtered_traces = []
    for trace, conf in zip(traces, confidences):
        if mode=="top" and conf >= threshold:
            filtered_traces.append(trace)
        elif mode=="bottom" and conf <= threshold:
            filtered_traces.append(trace)
    
    return filtered_traces, threshold

def compute_top_threshold_by1derivative(confs):
    sorted_values = np.sort(confs)
    
    if len(sorted_values) <= 1:
        return (sorted_values[0].item(),1) if len(sorted_values) == 1 else (0,0)
    
    max_change = 0
    max_change_conf = sorted_values[0]
    max_idx = 0

    for i in range(len(sorted_values) - 1):
        current_val = sorted_values[i]
        next_val = sorted_values[i + 1]
        
        if current_val != 0:
            relative_change = abs((next_val - current_val))
        else:
            relative_change = float('inf') if next_val != 0 else 0
        
        if relative_change > max_change:
            max_change = relative_change
            max_change_conf = next_val
            max_idx = i+2
    
    return max_change_conf, max_idx

def compute_top_threshold_by2derivative(confs, window_size=30):
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    sorted_values = np.sort(confs)
    sorted_values_np = sorted_values
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

def calculate_bottom_window_confidence(trace: Dict[str, Any], window_size: int = 2048, bottom_percent: float = 0.1) -> float:
    """Calculate mean confidence from sliding windows, return average of bottom percentile"""
    try:
        if "group_confs" in trace and trace["group_confs"]:
            confs = trace["group_confs"]
            window_means = confs
        elif 'confs' in trace and trace['confs']:
            confs = trace['confs']
            if len(confs) < window_size:
                return np.mean(confs)
            
            window_means = []
            current_sum = sum(confs[:window_size])
            window_means.append(current_sum / window_size)
            
            for i in range(1, len(confs) - window_size + 1):
                current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
                window_means.append(current_sum / window_size)
        elif "step_confs" in trace and trace["step_confs"]:
            confs = trace["step_confs"]
            window_means = [np.mean(conf) for conf in confs]
            
        if not window_means:
            return 0.0
        
        if bottom_percent == -1:  # Min window
            return min(window_means)
        
        num_bottom = max(1, int(len(window_means) * bottom_percent))
        if num_bottom == 1:
            return min(window_means)
        else:
            bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
            return np.mean(bottom_means)
        
        return 0.0
    except Exception:
        return 0.0

def compute_all_voting_results(traces: List[Dict[str, Any]], qid) -> Dict[str, Any]:
    """Compute results for all voting methods"""
    # Extract valid traces with answers
    valid_indices, valid_traces = zip(*[(idx,trace) for idx, trace in enumerate(traces) if trace.get('extracted_answer')])
    
    if not valid_traces:
        return {method: None for method in [
            'majority', 'mean_confidence_weighted', 'tail_confidence_weighted',
            'bottom_window_weighted', 'min_window_weighted', 
            'top10_tail_filtered', 'top10_bottom_window_filtered'
        ]}
    
    # Extract answers for voting
    answers = [trace['extracted_answer'] for trace in valid_traces]
    
    # Calculate different types of confidences
    mean_confidences = [calculate_mean_confidence(trace) for trace in valid_traces]
    tail_confidences = [calculate_tail_confidence(trace) for trace in valid_traces]
    bottom_window_confidences = [calculate_bottom_window_confidence(trace) for trace in valid_traces]
    min_window_confidences = [calculate_bottom_window_confidence(trace, bottom_percent=-1) for trace in valid_traces]
    
    voting_results = {}
    
    # 1. Simple majority vote
    majority_answer = simple_majority_vote(answers)
    voting_results['majority'] = {
        'answer': majority_answer,
        'num_votes': len(answers),
        'confidence': None
    }
    
    # 2. Mean confidence weighted vote
    if any(c > 0 for c in mean_confidences):
        mean_weighted_answer = weighted_majority_vote(answers, mean_confidences)
        voting_results['mean_confidence_weighted'] = {
            'answer': mean_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(mean_confidences)
        }
    
    # 3. Tail confidence weighted vote
    if any(c > 0 for c in tail_confidences):
        tail_weighted_answer = weighted_majority_vote(answers, tail_confidences)
        voting_results['tail_confidence_weighted'] = {
            'answer': tail_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(tail_confidences)
        }
    
    # 4. Bottom window confidence weighted vote
    if any(c > 0 for c in bottom_window_confidences):
        bottom_weighted_answer = weighted_majority_vote(answers, bottom_window_confidences)
        voting_results['bottom_window_weighted'] = {
            'answer': bottom_weighted_answer,
            'num_votes': len(answers),
            'confidence': np.mean(bottom_window_confidences)
        }
    
    # 5. Min window confidence weighted vote
    if any(c > 0 for c in min_window_confidences):
        min_window_answer = weighted_majority_vote(answers, min_window_confidences)
        voting_results['min_window_weighted'] = {
            'answer': min_window_answer,
            'num_votes': len(answers),
            'confidence': np.mean(min_window_confidences)
        }
    
    # 6. Top 10% tail confidence filtered + weighted vote
    top_tail_traces, _ = filter_top_confidence(valid_traces, 'tail', 0.1)
    if top_tail_traces:
        top_tail_answers = [trace['extracted_answer'] for trace in top_tail_traces]
        top_tail_confidences = [calculate_tail_confidence(trace) for trace in top_tail_traces]
        
        if any(c > 0 for c in top_tail_confidences):
            top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
            voting_results['top10_tail_filtered'] = {
                'answer': top_tail_answer,
                'num_votes': len(top_tail_answers),
                'confidence': np.mean(top_tail_confidences)
            }
    
    # 7. Top 10% bottom window confidence filtered + weighted vote
    top_bottom_traces, _ = filter_top_confidence(valid_traces, 'bottom_window', 0.1)
    if top_bottom_traces:
        top_bottom_answers = [trace['extracted_answer'] for trace in top_bottom_traces]
        top_bottom_confidences = [calculate_bottom_window_confidence(trace) for trace in top_bottom_traces]
        
        if any(c > 0 for c in top_bottom_confidences):
            top_bottom_answer = weighted_majority_vote(top_bottom_answers, top_bottom_confidences)
            voting_results['top10_bottom_window_filtered'] = {
                'answer': top_bottom_answer,
                'num_votes': len(top_bottom_answers),
                'confidence': np.mean(top_bottom_confidences)
            }
    
    # 8. Top 10% tail window confidence filtered + weighted vote
    traj_conf_type = "tail"
    dual_filter_traces = []
    while True:
        top_traces, top_threshold = filter_top_confidence(valid_traces, traj_conf_type, 0.1)
        bottom_traces, bottom_threshold = filter_top_confidence(valid_traces, traj_conf_type, 0.9, mode="bottom")
        if top_traces and bottom_traces:
            top_answers = [trace['extracted_answer'] for trace in top_traces]
            bottom_answers = [trace['extracted_answer'] for trace in bottom_traces]

            if traj_conf_type == "tail":
                top_confidences = [calculate_tail_confidence(trace) for trace in top_traces]
                bottom_confidences = [calculate_tail_confidence(trace) for trace in bottom_traces]
            elif traj_conf_type == "bottom":
                top_confidences = [calculate_bottom_window_confidence(trace) for trace in top_traces]
                bottom_confidences = [calculate_bottom_window_confidence(trace) for trace in bottom_traces]
            elif traj_conf_type == "mean":
                top_confidences = [calculate_mean_confidence(trace) for trace in top_traces]
                bottom_confidences = [calculate_mean_confidence(trace) for trace in bottom_traces]
            elif traj_conf_type == "lowest":
                top_confidences = [calculate_bottom_window_confidence(trace, bottom_percent=-1) for trace in top_traces]
                bottom_confidences = [calculate_bottom_window_confidence(trace, bottom_percent=-1) for trace in bottom_traces]

            if any(c > 0 for c in top_confidences):
                top_answer = weighted_majority_vote(top_answers, top_confidences)
            
            if any(c > 0 for c in bottom_confidences):
                bottom_answer = weighted_majority_vote(bottom_answers, bottom_confidences)
            
            dual_filter_traces.append({
                "top_answers": top_answers,
                "top_answer": top_answer,
                "top_confidences": top_confidences,
                "top_threshold": top_threshold,
                "bottom_answers": bottom_answers,
                "bottom_answer": bottom_answer,
                "bottom_confidences": bottom_confidences,
                "bottom_threshold": bottom_threshold,
                "valid_indices": valid_indices
            })
            if top_answer != bottom_answer:
                current_valid_indices, current_valid_traces = zip(*[(idx, trace) for idx, trace in enumerate(traces) if trace.get('extracted_answer') and trace['extracted_answer'] != bottom_answer])
                valid_indices = [idx for idx in current_valid_indices if idx in valid_indices]
                valid_traces = [traces[idx] for idx in valid_indices]
                
                # """"
                top_traces = filter_top_confidence(valid_traces, traj_conf_type, 0.1)
                if traj_conf_type == "tail":
                    top_confidences = [calculate_tail_confidence(trace) for trace in top_traces]
                elif traj_conf_type == "bottom":
                    top_confidences = [calculate_bottom_window_confidence(trace) for trace in top_traces]
                elif traj_conf_type == "mean":
                    top_confidences = [calculate_mean_confidence(trace) for trace in top_traces]
                elif traj_conf_type == "lowest":
                    top_confidences = [calculate_bottom_window_confidence(trace, bottom_percent=-1) for trace in top_traces]
                top_answer = weighted_majority_vote(top_answers, top_confidences)
                voting_results['dual_window_filtered'] = {
                    'answer': top_answer,
                    'num_votes': len(top_answers),
                    'confidence': np.mean(top_confidences)
                }
                break
                # """
            else:
                voting_results['dual_window_filtered'] = {
                    'answer': top_answer,
                    'num_votes': len(top_answers),
                    'confidence': np.mean(top_confidences)
                }
                break
    
    return voting_results, dual_filter_traces

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

def read_pkl_files(directory_path: str, pattern) -> Dict[str, Any]:
    pkl_data = {}

    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        return pkl_data

    # Find all pkl files in the directory
    pkl_files = natsorted(glob.glob(os.path.join(directory_path, pattern)))

    if not pkl_files:
        print(f"No pkl files found in {directory_path}")
        return pkl_data

    print(f"Found {len(pkl_files)} pkl files in {directory_path}")

    for pkl_file in tqdm(sorted(pkl_files)):
        filename = os.path.basename(pkl_file)
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            pkl_data[filename] = data

            # Get basic info about the data
            # data_type = type(data).__name__
            # if hasattr(data, '__len__'):
            #     data_size = len(data)
            #     print(f"  {filename}: {data_type} with {data_size} items")
            # else:
            #     print(f"  {filename}: {data_type}")

        except Exception as e:
            print(f"  Error reading {filename}: {e}")

    return pkl_data

def get_voting_traces(pkl_data):
    voting_traces = {}
    gt = {}

    for trace_data in pkl_data.values():
        qid = trace_data["qid"]
        traces = []
        for trace in trace_data["all_traces"]:
            traces.append(trace)
        
        voting_traces[qid]=traces
        gt[qid]=trace_data["ground_truth"]
    
    return voting_traces, gt

def get_step_traces(pkl_data):
    step_traces = {}
    
    for filename, trace_data in pkl_data.items():
        qid = int(filename.split('_')[-1].split('.')[0])
        trace_data["step"]["extracted_answer"] = max(trace_data["answer_count"])
        if not step_traces.get(qid):
            step_traces[qid] = [trace_data["step"]]
        else:
            step_traces[qid].append(trace_data["step"])
    return step_traces

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

ADAPTIVE_TYPE = 0
CONF_THRESHOLD = 16
if __name__ == "__main__":
    res_dir = "/share/yangxizhong/output/deepconf/stepconf_deepthink/gpqad-low-warmupK-parallel-20251117/voting_info_final"
    save_dir  = f"{res_dir}/voting_test"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{ADAPTIVE_TYPE}derivative_voting.json"
    pkl_files = natsorted(glob.glob(os.path.join(res_dir, "deepthink*.pkl")))
    # res_window_data = read_pkl_files(res_dir, "deepthink*.pkl")
    # res_step_data = read_pkl_files(res_dir, "logitsprocessor*.pkl")
    # warmup_traces, ground_truth = get_voting_traces(res_window_data)
    # step_traces = get_step_traces(res_step_data)

    # voting_traces = {key:warmup_traces.get(key, [])+step_traces.get(key, []) for key in warmup_traces}
    # voting_traces = warmup_traces

    res = []
    for pkl_path in tqdm(pkl_files):
        with open(pkl_path, "rb") as f:
            data=pickle.load(f)
        qid = data["qid"]
        if data.get("all_traces"):
            voting_trace = data["all_traces"]
        else:
            voting_trace = data["final_traces"]
        gt = data["ground_truth"]
        
        if should_skip_qid(qid, save_path):
            print(f"skip {qid}")
            continue
        results, dual_filter_traces = compute_all_voting_results(voting_trace, qid)
        for key, value in results.items():
            answer = value["answer"]
            correct = equal_func(answer, gt)
            results[key]["correct"] = correct
        
        for dual_filter_trace in dual_filter_traces:
            top_answer = dual_filter_trace["top_answer"]
            bottom_answer = dual_filter_trace["bottom_answer"]
            dual_filter_trace["top_answer_correctness"] = equal_func(top_answer, gt)
            dual_filter_trace["bottom_answer_correctness"] = equal_func(bottom_answer, gt)

        res = {
                "qid": qid,
                "evaluation_res": results,
                # "dual_filter_traces": dual_filter_traces,
            }
        save_detail_res2json(res, save_path)
    
    print("mix/multi stepconf voting successfully!")

    