import time
import copy
import numpy as np
from tqdm import tqdm
from math import ceil
from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams
from .wrapper import DeepThinkLLM
from .utils import process_batch_results
from .outputs import DeepThinkOutput

class AdaptiveSamplingDeepThinkLLM(DeepThinkLLM):
    def __init__(self, model: str, **vllm_kwargs):
        # Call parent class initialization
        super().__init__(model, **vllm_kwargs)
        self.answer_count = {}
        self.answer_weight = {}
    
    def _deepthink_online(
        self,
        prompt: str,
        output: DeepThinkOutput,
        warmup_traces: int,
        total_budget: int,
        confidence_percentile: int,
        window_size: int,
        sampling_params: Optional[SamplingParams]
    ) -> DeepThinkOutput:
    
        processing_start = time.time()
        
        # Warmup phase
        print(f"Starting warmup phase...", sampling_params)
        warmup_gen_start = time.time()
        # Generate warmup traces
        warmup_params_list = []
        base_seed = time.time_ns()
        for param_id in range(warmup_traces):
            warmup_params = copy.deepcopy(sampling_params) 
            warmup_params.logprobs = 20
            warmup_params.seed = base_seed + param_id
            warmup_params_list.append(warmup_params)
        warmup_outputs = self.llm.generate([prompt for _ in range(warmup_traces)], warmup_params_list)
        output.warmup_gen_time = time.time() - warmup_gen_start
        
        # Process warmup results
        warmup_process_start = time.time()
        warmup_result = process_batch_results(warmup_outputs, window_size)
        output.warmup_process_time = time.time() - warmup_process_start

        print('Warmup min_confs:', warmup_result['min_confs'])
        output.conf_bar = float(np.percentile(warmup_result['min_confs'],100 - confidence_percentile))
        output.warmup_min_confs = warmup_result['min_confs']
        
        output.warmup_traces = warmup_result['traces']
        output.warmup_tokens = warmup_result['total_tokens']
        print(f"Warmup completed: conf_bar={output.conf_bar:.3f}")

        # Final phase
        print(f"Starting final phase...", sampling_params)
        final_gen_start = time.time()

        final_params_list = []
        for param_id in range(total_budget - warmup_traces):
            final_params = copy.deepcopy(sampling_params) 
            final_params.logprobs = 20
            final_params.seed = base_seed + param_id + warmup_traces
            final_params.extra_args = {
                "conf_threshold": output.conf_bar,
                "eos_token_id": self.tokenizer.eos_token_id,
                "conf_group_size": window_size,
                "conf_topk": 20,
            }
            final_params_list.append(final_params)
        
        final_outputs = []
        final_result = {}
        batch_num = ceil((total_budget - warmup_traces)/warmup_traces)
        for batch_idx in tqdm(range(batch_num)):
            start_idx = batch_idx * warmup_traces
            end_idx = min(start_idx + warmup_traces, total_budget - warmup_traces)
            batch_prompts = [prompt for _ in range(end_idx - start_idx)]
            batch_params = final_params_list[start_idx:end_idx]

            batch_outputs = self.llm.generate(batch_prompts, batch_params)
            final_outputs.extend(batch_outputs)
            stop, batch_results = self.adaptive_stop(batch_outputs, window_size, output.conf_bar)

            if batch_idx == 0 :
                final_result = batch_results
            else:
                for key in batch_results.keys():
                    if type(batch_results[key]) == list:
                        final_result[key].extend(batch_results[key])
                    elif type(batch_results[key]) == int:
                        final_result[key] += batch_results[key]

            if stop:
                break

        # final_outputs = self.llm.generate([prompt for _ in range(total_budget - warmup_traces)], final_params_list)
        output.final_gen_time = time.time() - final_gen_start
        
        # Process final results
        # final_process_start = time.time()
        # final_result = process_batch_results(final_outputs, window_size)
        # output.final_process_time = time.time() - final_process_start
        
        print('Final min_confs:', final_result['min_confs'])
        output.final_min_confs = final_result['min_confs']
        
        output.final_traces = final_result['traces']
        output.final_tokens = final_result['total_tokens']
        
        # Apply confidence threshold to final traces
        for trace in output.final_traces:
            if trace["min_conf"] < output.conf_bar:
                trace["stop_reason"] = "gconf_threshold"
        
        # Combine all traces
        output.all_traces = output.warmup_traces + output.final_traces
        output.total_tokens = output.warmup_tokens + output.final_tokens
        output.total_traces_count = len(output.all_traces)
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        return output
    
    def adaptive_stop(self, batch_outputs, window_size, conf_bar, threshold=0.95):
        batch_result = process_batch_results(batch_outputs, window_size)
        
        for idx in range(len(batch_result["traces"])):
            answer = batch_result["traces"][idx]["extracted_answer"]
            conf_weight = batch_result["min_confs"][idx]
            if answer and conf_weight >= conf_bar:
                if answer not in self.answer_count:
                    self.answer_count[answer] = 1
                    self.answer_weight[answer] = conf_weight
                else:
                    self.answer_count[answer] += 1
                    self.answer_weight[answer] += conf_weight

        if self.answer_count:
            majority_answer = max(self.answer_count, key=self.answer_count.get)
            beta = self.answer_weight[majority_answer]/sum(self.answer_weight.values())
        
            if beta >= threshold and len(self.answer_count)>=2:
                stop = True
                print(f"early stop, beta={beta}")
            else:
                stop = False
        else:
            stop = False
        
        return stop, batch_result