import re
import os
import json
import glob
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dynasor.core.evaluator import math_equal
from sklearn.mixture import GaussianMixture
from multiprocessing import Pool
import functools
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# 添加GPU相关的全局变量
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()

def simple_majority_vote(answers: List[str]) -> Optional[str]:
    """Simple majority voting"""
    if not answers:
        return None
    
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
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
    if MOB_REWARD:
        sorted_indices_selctions_probs *= np.array(rewards)[sorted_indices]
    for i in range(len(unique_final_answers)):
        final_ans = unique_final_answers[i]
        final_ans_sorted_indices = np.where((final_answers[sorted_indices] == final_ans))[0]
        dist[i] = np.sum(sorted_indices_selctions_probs[final_ans_sorted_indices])
    return dist, unique_final_answers

def mob_adaptive_m(final_answers, 
                   rewards,
                   mob_q=0.1,
                   return_m_value = False,
                   **kwargs):
    # return weighted_majority_vote(final_answers, rewards)
    if "interval" in VOTING_MODE:
        return mob_interval(final_answers, rewards)

    if not MOB_ADAPTIVE:
        return mob_poly_m(final_answers, rewards)

    final_answers = np.array(final_answers)
    n = len(final_answers)
    if n == 1:
        return str(final_answers[0])
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

def mob_interval(final_answers, rewards, interval=0.1):
    interval = INTERVAL
    rewards = np.array(rewards)
    sorted_indices = np.argsort(rewards)[::-1]
    sorted_answers = np.array(final_answers)[sorted_indices]
    sorted_rewards = np.array(rewards)[sorted_indices]
    
    rewards_scope = max(sorted_rewards) - min(sorted_rewards)
    step_length = rewards_scope*interval
    bon_answers = [[]]
    bon_rewards = [[]]
    
    rewards_interval_upper = max(sorted_rewards) - step_length
    for idx in range(len(sorted_rewards)):
        if sorted_rewards[idx] >= rewards_interval_upper:
            bon_answers[-1].append(sorted_answers[idx])
            bon_rewards[-1].append(sorted_rewards[idx])
        else:
            bon_answers.append([sorted_answers[idx]])
            bon_rewards.append([sorted_rewards[idx]])
            rewards_interval_upper -= step_length
    
    best_answers = []
    for idx in range(len(bon_answers)):
        most_common_answer = str(Counter(bon_answers[idx]).most_common(1)[0][0])
        best_answers.append(most_common_answer)

    final_answer = weighted_majority_vote(best_answers, [float(np.mean(reward)) for reward in bon_rewards])

    return final_answer


def calculate_tail_confidence(trace: Dict[str, Any], tail_tokens: int = 2048) -> float:
    """Calculate mean confidence from the last N tokens"""
    try:
        if "window_confs" in trace and trace["window_confs"]:
            confs = trace["window_confs"]
            if type(confs) == list:
                tail_confs = confs[-1]
            else:
                tail_confs = [confs]
            return np.mean(tail_confs) if tail_confs else 0.0
        elif "group_confs" in trace and trace["group_confs"]:
            confs = trace["group_confs"]
            if type(confs) == list:
                tail_confs = confs[-1]
            else:
                tail_confs = [confs]
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

def gmm_calculate_conf_threshold(confidences):
    data = np.array(confidences).reshape(-1, 1)
    if len(data)<=1:
        return 0
    
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(data)
    labels = gmm.predict(data)
    dist1 = data[labels == 0]
    dist2 = data[labels == 1]
    
    if len(dist1)==0:
        conf_bar = dist2.min()
    elif len(dist2)==0:
        conf_bar = dist1.min()
    elif dist1.max() <= dist2.min():
        conf_bar = (dist1.max() + dist2.min())/2
    else:
        conf_bar = (dist2.max() + dist1.min())/2
    
    return conf_bar

def filter_top_confidence(traces: List[Dict[str, Any]], confidence_type: str = 'tail', top_percent: float = 0.1, mode="top", threshold=None, gt=None) -> List[Dict[str, Any]]:
    """Filter traces by top confidence percentage"""
    if not traces:
        return []
    
    # Calculate confidences
    confidences = []
    for trace in traces:
        if confidence_type == 'tail':
            conf = calculate_tail_confidence(trace)
        if trace.get("correctness", False):
            confidences.append(conf)
        else:
            confidences.append(conf)
    
    # Get threshold for top percentage
    if not threshold:
        if THRESHOLD_MODE == "top10":
            threshold = np.percentile(confidences, (1 - top_percent) * 100)
        elif THRESHOLD_MODE == "gmm":
            threshold = gmm_calculate_conf_threshold(confidences)
    
    # Filter traces
    filtered_traces = []
    for trace, conf in zip(traces, confidences):
        if mode=="top" and conf >= threshold:
            filtered_traces.append(trace)
        elif mode=="bottom" and conf <= threshold:
            filtered_traces.append(trace)
    
    return filtered_traces, confidences, threshold


def adaptive_stop(answers, confidences, beta=0.95):
    most_common_answer = simple_majority_vote(answers)
    consistency = sum([confidences[idx] for idx, ans in enumerate(answers) if ans==most_common_answer])/sum(confidences)
    if consistency >= beta and len(answers)>=2:
        stop = True
    else:
        stop = False
    
    return stop

def compute_all_voting_results(traces: List[Dict[str, Any]], qid, all_threshold=None, gt=None) -> Dict[str, Any]:
    """Compute results for all voting methods"""
    # Extract valid traces with answers
    valid_traces = [trace for trace in traces if trace.get('extracted_answer')]
    
    if not valid_traces:
        return {
            'answer': None,
            'num_votes': 0,
            'confidence': None,
            'top10_threshold': 0
        }
    
    if "reject" not in VOTING_MODE.lower():
        if THRESHOLD_MODE:
            top_tail_traces, _, top10_threshold = filter_top_confidence(valid_traces, confidence_type='tail', top_percent=TOP_PERCENT, threshold=all_threshold)
            top_tail_answers = [trace['extracted_answer'] for trace in top_tail_traces]
            top_tail_confidences = [calculate_tail_confidence(trace) for trace in top_tail_traces]

            if top_tail_traces:
                if "wsc" in VOTING_MODE.lower():
                    top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
                elif "hierv" in VOTING_MODE.lower():
                    top_tail_answer = mob_interval(top_tail_answers, top_tail_confidences)
            else:
                top_tail_answer = None

        else:
            top_tail_answers = [trace['extracted_answer'] for trace in valid_traces]
            top_tail_confidences = [calculate_tail_confidence(trace) for trace in valid_traces]

            if "wsc" in VOTING_MODE.lower():
                top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
            elif "hierv" in VOTING_MODE.lower():
                top_tail_answer = mob_interval(top_tail_answers, top_tail_confidences)
            elif VOTING_MODE == "SC":
                top_tail_answer = simple_majority_vote(top_tail_answers)
            elif VOTING_MODE == "BoN":
                max_conf_idx = top_tail_confidences.index(max(top_tail_confidences))
                best_ans = top_tail_answers[max_conf_idx]
                top_tail_answer = best_ans
            elif VOTING_MODE == "MoB-Adaptive":
                top_tail_answer = mob_adaptive_m(top_tail_answers, top_tail_confidences)
            top10_threshold = 0
    else:
        top_tail_traces, _, top10_threshold = filter_top_confidence(valid_traces, confidence_type='tail', top_percent=TOP_PERCENT, threshold=all_threshold)
        top_tail_answers = [trace['extracted_answer'] for trace in top_tail_traces]
        top_tail_confidences = [calculate_tail_confidence(trace) for trace in top_tail_traces]

        if top_tail_traces:
            bottom_tail_traces, _, _ = filter_top_confidence(valid_traces, confidence_type='tail', top_percent=TOP_PERCENT, threshold=all_threshold, mode="bottom")
            if bottom_tail_traces:
                bottom_tail_answers = [trace['extracted_answer'] for trace in bottom_tail_traces]
                bottom_tail_confidences = [-calculate_tail_confidence(trace) for trace in bottom_tail_traces]
                if "wsc" in VOTING_MODE.lower():
                    bottom_tail_answer = weighted_majority_vote(bottom_tail_answers, bottom_tail_confidences)
                    top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
                elif "hierv" in VOTING_MODE.lower():
                    bottom_tail_answer = mob_interval(bottom_tail_answers, bottom_tail_confidences)
                    top_tail_answer = mob_interval(top_tail_answers, top_tail_confidences)
                
                if bottom_tail_answer != top_tail_answer:
                    valid_traces = [trace for trace in valid_traces if trace["extracted_answer"] != bottom_tail_answer]
                    if valid_traces:
                        top_tail_traces, _, top10_threshold = filter_top_confidence(valid_traces, confidence_type='tail', top_percent=TOP_PERCENT, threshold=all_threshold)
                        top_tail_answers = [trace['extracted_answer'] for trace in top_tail_traces]
                        top_tail_confidences = [calculate_tail_confidence(trace) for trace in top_tail_traces]
                        if "wsc" in VOTING_MODE.lower():
                            top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
                        elif "hierv" in VOTING_MODE.lower():
                            top_tail_answer = mob_interval(top_tail_answers, top_tail_confidences)
            else:
                if "wsc" in VOTING_MODE.lower():
                    top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
                elif "hierv" in VOTING_MODE.lower():
                    top_tail_answer = mob_interval(top_tail_answers, top_tail_confidences)
        else:
            top_tail_answer = None
    
    if top_tail_answer:
        voting_results = {
            'answer': top_tail_answer,
            'num_votes': len(top_tail_answers),
            'confidence': np.mean(top_tail_confidences),
            'top10_threshold': top10_threshold,
        }
    else:
        voting_results = {
            'answer': None,
            'num_votes': 0,
            'confidence': None,
            'top10_threshold': top10_threshold
        }
    return voting_results

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
    if not answer:
        return False
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)

def dict_to_xlsx_dynamic(data_dict, output_file="results.xlsx", update_mode="merge"):
    """
    动态更新Excel文件
    
    Args:
        data_dict: {method: {benchmark: score, ...}, ...}
        output_file: 输出文件名
        update_mode: "append", "overwrite", "merge"
    """
    # 创建新的DataFrame
    new_df = pd.DataFrame.from_dict(data_dict, orient='index')
    new_df.reset_index(inplace=True)
    new_df.rename(columns={'index': 'Method'}, inplace=True)
    
    # 检查文件是否存在
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_excel(output_file)
            
            if update_mode == "append":
                # 简单追加
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            elif update_mode == "merge":
                # 基于Method列合并，新数据覆盖旧数据
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
                    subset=['Method'], keep='last'
                ).reset_index(drop=True)
            
            elif update_mode == "overwrite":
                # 完全覆盖
                combined_df = new_df
            
            else:
                raise ValueError(f"不支持的更新模式: {update_mode}")
                
        except Exception as e:
            print(f"读取现有文件失败: {e}，将创建新文件")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # 保存更新后的数据
    combined_df.to_excel(output_file, index=False)
    print(f"结果已动态更新到: {output_file}")
    
    return combined_df

def should_skip_exp(exps_name, save_path):
    if os.path.exists(save_path):
        existing_df = pd.read_excel(save_path)
        if exps_name in list(existing_df.Method):
            return True
        else:
            return False

def get_all_threshold(pkl_files):
    """获取所有阈值的函数 - 需要根据你的实际实现补充"""
    # 这里需要根据你的实际代码实现
    # 返回 all_traces, all_threshold
    return {}, None

# 将需要在多进程中使用的函数移到全局作用域
def process_single_question_worker(args):
    """全局函数，用于多进程处理单个问题"""
    qid, pkl_path, all_threshold = args
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    qid = data["qid"]
    if data.get("all_traces"):
        voting_trace = data["all_traces"]
    else:
        voting_trace = data["final_traces"]
    gt = data["ground_truth"]
    
    # 为每个问题生成64个子采样
    all_results = []
    
    sub_trace = voting_trace
    result = compute_all_voting_results(sub_trace, qid, all_threshold)
    answer = result["answer"]
    correct = equal_func(answer, gt) if answer else False
    result["correct"] = correct
    all_results.append(result)
    
    return all_results

def simple_parallel_process(pkl_files: List[str], 
                           all_threshold=None, 
                           num_workers: int = 4) -> List[List[Dict]]:
    """简化的并行处理，避免复杂的GPU操作在多进程中"""
    
    # 使用线程池而不是进程池
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(qid, pkl_path, all_threshold) for qid, pkl_path in enumerate(pkl_files)]
        results = list(executor.map(process_single_question_worker, args_list))
    
    return results

def optimized_process(exps, res_dir_dict, save_path):
    """优化后的主处理函数"""
    final_results = {}
    final_infos = {}
    
    for exp_name, exp_params in tqdm(exps.items()):
        global MODE, MOB_ADAPTIVE, MOB_REWARD, VOTING_MODE, THRESHOLD_MODE, THRESHOLD_SCOPE, EXP_NAME
        # EXP_NAME = exp_name
        EXP_NAME = ""
        if MODE == 1 or MODE == 2 or MODE == 3:
            MOB_ADAPTIVE, MOB_REWARD, VOTING_MODE, THRESHOLD_MODE, THRESHOLD_SCOPE = exp_params
            save_pkl_name = "tmp.pkl"
        elif MODE == 0:
            MOB_ADAPTIVE, MOB_REWARD, VOTING_MODE, THRESHOLD_MODE, THRESHOLD_SCOPE, save_pkl_name = exp_params
        
        print(f"Processing {exp_name} with params {exp_params}")
        
        if "while" in exp_name.lower():
            print(f"skip while: {exp_name}")
            continue
            
        if should_skip_exp(exp_name, save_path):
            print(f"skip exists: {exp_name}")
            continue
        
        final_results[exp_name] = {}
        final_infos[exp_name] = {}
        
        for benchmark, res_dir in res_dir_dict.items():
            if not os.path.isdir(res_dir):
                final_results[exp_name][benchmark] = 0
                final_infos[exp_name][benchmark] = 0
                continue
                
            save_dir = f"{res_dir}/voting_test"
            pkl_files = natsorted(glob.glob(os.path.join(res_dir, "deepthink*.pkl")))
            os.makedirs(save_dir, exist_ok=True)
            
            if MODE == 2:
                # 使用优化后的并行处理
                print(f"Processing {len(pkl_files)} files with threading...")
                
                # 设置工作线程数
                num_workers = min(8, mp.cpu_count())
                
                total_res = simple_parallel_process(pkl_files, None, num_workers)
                
                # 计算准确率
                voting_acc = sum([one_data["correct"] for question_data in total_res for one_data in question_data]) / (len(total_res) * 64)
                print(f"{benchmark} voting acc: {voting_acc}")
                
                final_results[exp_name][benchmark] = voting_acc
                final_infos[exp_name][benchmark] = total_res
        
        dict_to_xlsx_dynamic(final_results, save_path)
        
        # 清理GPU缓存
        if USE_GPU:
            torch.cuda.empty_cache()
    
    return final_results, final_infos

if __name__ == "__main__":
    global MODE, MOB_ADAPTIVE, MOB_REWARD, VOTING_MODE, THRESHOLD_MODE, THRESHOLD_SCOPE, TOP_PERCENT, INTERVAL
    MODE = 2 # 0:allthreshold allmax, 1:norm test and allTop test, 2: multi avg voting, 3: ALLmax
    MOB_ADAPTIVE=1 # adaptive m of mob or N^(0.5)
    MOB_REWARD=0  # mob prob * rewards
    VOTING_MODE="mob" # mob, weighted, mob_interval_simple, mob_interval_meanweighted, mob_reject, mob_reject_whileAll, mob_reject_whileFinal, mob_intervalgmm_simple, mob_intervalgmm_meanweighted, # mob, weighted, mob_interval_simple, mob_interval_meanweighted, mob_reject, mob_reject_whileAll, mob_reject_whileFinal, mob_intervalabs_simple, mob_intervalabs_meanweighted
    THRESHOLD_MODE = "gmm"  # top10, gmm
    THRESHOLD_SCOPE = "all" # single, all
    TOP_PERCENT = 0.5
    GMM_COMPARE = 1
    INTERVAL = 0.1

    # model_list = ["dpsk-distill-qwen3-8b", "Qwen3-8B", "Qwen3-14B", "Qwen3-14B-NonThinking", "Qwen3-32B"]
    model_list = ["Qwen3-0.6B-Thinking", "Qwen3-0.6B-NonThinking", "Qwen3-1.7B-Thinking", "Qwen3-1.7B-NonThinking", "Qwen3-4B-Thinking", "Qwen3-4B-NonThinking", "Qwen3-8B-NonThinking", "Qwen3-32B-NonThinking", "dpsk-distill-qwen3-7b", "Llama-3.1-8B-Instruct", "Qwen2.5-Math-7B", "Qwen3-14B-NonThinking"]
    model_list = ["Qwen3-8B-NonThinking", "Qwen3-32B-NonThinking", "dpsk-distill-qwen3-7b", "Llama-3.1-8B-Instruct", "Qwen2.5-Math-7B", "Qwen3-14B-NonThinking"]
    for model_name in model_list:
        res_dir_dict = {
            "B512": {
                "res_dir_dict_base" : {
                    "hmmt2025": f"/share/yangxizhong/output/deepconf/baseline-dpsk/hmmt2025-B512/{model_name}/voting_info/only_save_tail_confs",
                    "gpqad": f"/share/yangxizhong/output/deepconf/baseline-dpsk/gpqad-B512/{model_name}/voting_info/only_save_tail_confs",
                    "aime2024": f"/share/yangxizhong/output/deepconf/baseline-dpsk/aime2024-B512/{model_name}/voting_info/only_save_tail_confs",
                    "aime2025": f"/share/yangxizhong/output/deepconf/baseline-dpsk/aime2025-B512/{model_name}/voting_info/only_save_tail_confs",
                    "brumo2025": f"/share/yangxizhong/output/deepconf/baseline-dpsk/brumo2025-B512/{model_name}/voting_info/only_save_tail_confs"
                },

                "res_dir_dict_selfstepconf" : {
                    "hmmt2025_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B512-95-80-80-20251207/{model_name}/voting_info/only_save_tail_confs",
                    "gpqad_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/gpqad-B512-95-80-80-20251204/{model_name}/voting_info/only_save_tail_confs",
                    "aime2024_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2024-B512-95-80-80-20251204/{model_name}/voting_info/only_save_tail_confs",
                    "aime2025_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2025-B512-95-80-80-20251204/{model_name}/voting_info/only_save_tail_confs",
                    "brumo2025_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/brumo2025-B512-95-80-80-20251204/{model_name}/voting_info/only_save_tail_confs",
                }
            },
            "B64": {
                "res_dir_dict_base" : {
                    "hmmt2025": f"/share/yangxizhong/output/deepconf/baseline-dpsk/hmmt2025-B64/{model_name}/voting_info/only_save_tail_confs",
                    "gpqad": f"/share/yangxizhong/output/deepconf/baseline-dpsk/gpqad-B64/{model_name}/voting_info/only_save_tail_confs",
                    "aime2024": f"/share/yangxizhong/output/deepconf/baseline-dpsk/aime2024-B64/{model_name}/voting_info/only_save_tail_confs",
                    "aime2025": f"/share/yangxizhong/output/deepconf/baseline-dpsk/aime2025-B64/{model_name}/voting_info/only_save_tail_confs",
                    "brumo2025": f"/share/yangxizhong/output/deepconf/baseline-dpsk/brumo2025-B64/{model_name}/voting_info/only_save_tail_confs"
                },

                "res_dir_dict_selfstepconf" : {
                    "hmmt2025_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-B64-95-80-80-20251204/{model_name}/voting_info/only_save_tail_confs",
                    "gpqad_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/gpqad-B64-95-80-80-20251202/{model_name}/voting_info/only_save_tail_confs",
                    "aime2024_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2024-B64-95-80-80-20251203/{model_name}/voting_info/only_save_tail_confs",
                    "aime2025_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2025-B64-95-80-80-20251203/{model_name}/voting_info/only_save_tail_confs",
                    "brumo2025_ssc": f"/share/yangxizhong/output/deepconf/selfstepconf_deepthink/brumo2025-B64-95-80-80-20251204/{model_name}/voting_info/only_save_tail_confs",
                }
            }
        }

        BUDGET = 63
        REPET = 64
        SUFFIX = "repeat3"
        res_dir_dicts = [
            [res_dir_dict["B64"]["res_dir_dict_base"], f"/share/yangxizhong/workspace/step_level_deepconf/output/result/voting/{model_name}-core-All-Avg{REPET}ofB{BUDGET}-AllAblation-base-{SUFFIX}.xlsx"],
            [res_dir_dict["B64"]["res_dir_dict_selfstepconf"], f"/share/yangxizhong/workspace/step_level_deepconf/output/result/voting/{model_name}-core-All-Avg{REPET}ofB{BUDGET}-AllAblation-ssc-{SUFFIX}.xlsx"],
        ]
             
        if MODE==1 or MODE==2 or MODE==3:
            final_results = {}
            final_infos = {}
            for model_idx in range(2):
                res_dir_dict, ori_save_path = res_dir_dicts[model_idx]
                save_path = ori_save_path  
                exps = {
                    "SC":                       (0,0,    "SC",                   None,       None),
                    "BoN":                      (0,0,   "BoN",                  None,       None),
                    "MoB-Adaptive":             (1,0,   "MoB-Adaptive",         None,       None),

                    "0000-Top50-No-WSC":        (0,0,   "Top50-No-WSC",         "top10",    None),
                    "0100-GMM-No-WSC":          (0,0,   "GMM-No-WSC",           "gmm",      None),
                    "0010-Top50-Reject-WSC":    (0,0,   "Top50-Reject-WSC",     "top10",    None),
                    "0001-Top50-No-HierV":      (0,0,   "Top50-No-HierV",       "top10",    None),
                    "0110-GMM-Reject-WSC":      (0,0,   "GMM-Reject-WSC",       "gmm",      None),
                    "0101-GMM-No-HierV":        (0,0,   "GMM-No-HierV",         "gmm",      None),
                    "0011-Top50-Reject-HierV":  (0,0,   "Top50-Reject-HierV",   "top10",    None),
                    "0111-GMM-Reject-HierV":    (0,0,   "GMM-Reject-HierV",     "gmm",      None),

                    "0000-No-No-WSC":           (0,0,   "No-No-WSC",            None,    None),
                    "0001-No-No-HierV":         (0,0,   "No-No-HierV",          None,    None),
                }
                        
                print(f"VOTING_MODE: {VOTING_MODE}, THRESHOLD_MODE: {THRESHOLD_MODE}, THRESHOLD_SCOPE: {THRESHOLD_SCOPE}")
                
                mode_result, mode_info = optimized_process(exps, res_dir_dict, save_path)
                final_results.update(mode_result)
                if mode_info:
                    final_infos.update(mode_info)
                if final_infos:
                    pkl_save_path = save_path.replace("xlsx", "pkl")
                    if os.path.exists(pkl_save_path):
                        with open(pkl_save_path, "rb") as f:
                            previous_final_info = pickle.load(f)
                        final_infos.update(previous_final_info)
                    with open(pkl_save_path, "wb") as f:
                        pickle.dump(final_infos, f)