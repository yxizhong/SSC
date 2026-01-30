"""
Utility functions for DeepThinkLLM

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dynasor.core.evaluator import math_equal

def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    
    return None

def compute_confidence(logprobs: List[Dict]) -> List[float]:
    """Compute confidence score from logprobs and return only confidence values"""
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns a dict of {token_id: Logprob object}
            # Get the selected token's logprob (the one with highest probability)
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs


def compute_least_grouped(confs: List[float], group_size: int) -> List[float]:
    """Compute sliding window mean confidence"""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    
    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means

def compute_least_step(tokens, confs, step_token_ids) -> List[float]:
    step_means = []

    start_idx = 0
    end_idx = 0
    for idx, token in enumerate(tokens):
        if token in step_token_ids:
            end_idx = idx
            window = confs[start_idx:end_idx]
            step_means.append(round(sum(window) / len(window), 3))
            start_idx = end_idx
    
    if not step_means:  # if don't have '\n\n'
        step_means.append(round(sum(confs) / len(confs), 3))

    return step_means

# ============= VOTING FUNCTIONS =============

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


def calculate_mean_confidence(trace: Dict[str, Any]) -> float:
    """Calculate mean confidence from confs in a trace"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            return np.mean(confs) if confs else 0.0
        return 0.0
    except Exception:
        return 0.0


def calculate_tail_confidence(trace: Dict[str, Any], tail_tokens: int = 2048) -> float:
    """Calculate mean confidence from the last N tokens"""
    try:
        if "group_confs" in trace and trace["group_confs"]:
            confs = trace["group_confs"]
            tail_confs = confs[-1]
            return tail_confs
        elif 'confs' in trace and trace['confs']:
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except Exception:
        return 0.0


def calculate_bottom_window_confidence(trace: Dict[str, Any], window_size: int = 2048, bottom_percent: float = 0.1) -> float:
    """Calculate mean confidence from sliding windows, return average of bottom percentile"""
    try:
        if "group_confs" in trace and trace["group_confs"]:
            confs = trace["group_confs"]
            window_means = [np.mean(conf) for conf in confs]

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


def filter_top_confidence(traces: List[Dict[str, Any]], confidence_type: str = 'tail', top_percent: float = 0.1) -> List[Dict[str, Any]]:
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
    threshold = np.percentile(confidences, (1 - top_percent) * 100)
    
    # Filter traces
    filtered_traces = []
    for trace, conf in zip(traces, confidences):
        if conf >= threshold:
            filtered_traces.append(trace)
    
    return filtered_traces


def compute_all_voting_results(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute results for all voting methods"""
    # Extract valid traces with answers
    valid_traces = [trace for trace in traces if trace.get('extracted_answer')]
    
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
    top_tail_traces = filter_top_confidence(valid_traces, 'tail', 0.1)
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
    top_bottom_traces = filter_top_confidence(valid_traces, 'bottom_window', 0.1)
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
    
    return voting_results


# ============= OUTPUT PROCESSING =============

def process_output(output, window_size: int) -> Dict[str, Any]:
    """Process a single vLLM output - for online mode with sliding window confidence"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence but don't store logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    sliding_window = compute_least_grouped(confs, group_size=window_size) if confs else [0]
    
    extracted_answer = extract_answer(text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # Store individual token confidences
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
        "extracted_answer": extracted_answer,
    }

def process_output_bystep(output, step_token_ids, entropy_bar=None) -> Dict[str, Any]:
    """Process a single vLLM output - for online mode with sliding window confidence"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence but don't store logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    sliding_window = compute_least_step(token_ids, confs, step_token_ids) if confs else [0]
    
    extracted_answer = extract_answer(text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # Store individual token confidences
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
        "extracted_answer": extracted_answer,
    }

def process_batch_results_bystep(batch_outputs, step_token_ids, entropy_bar=None) -> Dict[str, Any]:
    """Process batch results from vLLM for a single question"""
    #print('Osize: ', len(batch_outputs))
    question_outputs = []
    for output_list in batch_outputs:
        question_outputs += output_list.outputs
    # question_outputs = batch_outputs[0].outputs
    
    # Process all traces for this question
    traces = []
    min_confs = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output_bystep(output, step_token_ids, entropy_bar)
        traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'min_confs': min_confs,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }

def process_batch_results(batch_outputs, window_size: int) -> Dict[str, Any]:
    """Process batch results from vLLM for a single question"""
    #print('Osize: ', len(batch_outputs))
    question_outputs = []
    for output_list in batch_outputs:
        question_outputs += output_list.outputs
    # question_outputs = batch_outputs[0].outputs
    
    # Process all traces for this question
    traces = []
    min_confs = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output(output, window_size)
        traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'min_confs': min_confs,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }


def process_output_offline(output, window_size: int) -> Dict[str, Any]:
    """Process a single vLLM output for offline mode - stores full confidence array"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence but don't store full logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    
    extracted_answer = extract_answer(text)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # Store full confidence array for offline analysis
        "extracted_answer": extracted_answer,
    }


def process_batch_results_offline(batch_outputs, window_size: int) -> Dict[str, Any]:
    """Process batch results from vLLM for offline mode"""
    # question_outputs = batch_outputs[0].outputs
    question_outputs = []
    for output_list in batch_outputs:
        question_outputs += output_list.outputs

    # Process all traces for this question
    traces = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output_offline(output, window_size)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }

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