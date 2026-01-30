import os
import time
import copy
import glob
import pickle
import numpy as np
from tqdm import tqdm
from math import ceil
from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams
from .wrapper import DeepThinkLLM
from .utils import process_batch_results_bystep, compute_least_step
from .outputs import DeepThinkOutput
from .selfstepconf_processors import WrappedPerReqLogitsProcessor

class SelfStepConfDeepThinkLLM(DeepThinkLLM):
    def __init__(self, model: str, **vllm_kwargs):
        # Call parent class initialization
        super().__init__(model, processors=WrappedPerReqLogitsProcessor, **vllm_kwargs)
        self.step_tokens = ["\n"]. # "\n\n", "\n"
        # self.step_tokens = []
        self.reflection_token = "Hmm" # "wait, Wait, Hmm, Alternatively"
    
    def _deepthink_selfstepconf(
        self,
        prompt: str,
        output: DeepThinkOutput,
        total_budget,
        sampling_params: Optional[SamplingParams],
        save_path: str,
        extra_args=None,
        warmup_traces_dir=None
    ) -> DeepThinkOutput:
    
        processing_start = time.time()

        # Final phase
        print(f"Starting final phase...", sampling_params)
        final_gen_start = time.time()
        
        base_seed = time.time_ns()
        base_seed = 20251103

        final_params_list = []
        need_warmup = self.load_warmup_traces(output, save_path, warmup_traces_dir)
        if not need_warmup:
            total_budget = total_budget
        else:
            conf_bar_list = list(set(float(np.percentile(output.warmup_min_confs[:idx+1], 100 - 10)) for idx in range(len(output.warmup_min_confs))))
            total_budget = len(conf_bar_list)

        for param_id in range(total_budget):
            final_params = copy.deepcopy(sampling_params) 
            final_params.logprobs = 20
            final_params.seed = base_seed + param_id
            if total_budget == 1:
                final_save_path = save_path
            else:
                final_save_path = save_path.split('.pkl')[0] + f"_{param_id+1}" + ".pkl"
            
            finl_extra_args = copy.deepcopy(extra_args) 
            if need_warmup:
                finl_extra_args["conf_bar"] = conf_bar_list[param_id]
            else:
                finl_extra_args["conf_bar"] = 0.0

            final_params.extra_args = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "step_token_ids": [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens],
                "reflection_token_ids": self.tokenizer.encode(self.reflection_token, add_special_tokens=False)[0],
                "conf_topk": 20,
                "model_name": self.model_name,
                "save_path": final_save_path,
                "extra_args": finl_extra_args,
            }
            final_params_list.append(final_params)

        final_outputs = self.llm.generate([prompt for _ in range(total_budget)], final_params_list)
        output.final_gen_time = time.time() - final_gen_start
        
        # Process final results
        final_process_start = time.time()
        final_result = process_batch_results_bystep(final_outputs, [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens])
        output.final_process_time = time.time() - final_process_start
        
        print('Final min_confs:', final_result['min_confs'])
        output.final_min_confs = final_result['min_confs']
        
        output.final_traces = final_result['traces']
        output.final_tokens = final_result['total_tokens']
        
        # Combine all traces
        output.all_traces = output.warmup_traces + output.final_traces
        output.total_tokens = output.warmup_tokens + output.final_tokens
        output.total_traces_count = len(output.all_traces)
        
        # Basic voting (for backward compatibility)
        self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start
        return output
    
    def load_warmup_traces(self, output, save_path, warmup_traces_dir):
        qid = save_path.split('_')[-1].split('.')[0]
        pattern_path = os.path.join(warmup_traces_dir, f"deepthink_online_baseline_qid{qid}_*.pkl")
        file_paths = glob.glob(pattern_path)
        
        if len(file_paths) != 0:
            file_path = file_paths[0]
        else:
            return False
        print(f"load warmup traces from {file_path}")
        try:
            # Try torch.load first for CUDA tensors, fallback to pickle
            try:
                import torch
                import pickle
                data = torch.load(file_path, map_location='cpu')
            except:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False

        output.warmup_gen_time = data["timing_stats"]["warmup_gen_time"]
        output.warmup_process_time = data["timing_stats"]["warmup_process_time"]
        output.conf_bar = data["conf_bar"]
        if len(data["warmup_min_confs"]) == 0:
            output.warmup_min_confs = [min(compute_least_step(trace["token_ids"], trace["confs"], [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens])) for trace in data["all_traces"]] 
        else:
            output.warmup_min_confs = data["warmup_min_confs"]
        if len(data["warmup_traces"]) == 0:
            output.warmup_traces = data["all_traces"]
        else:
            output.warmup_traces = data["warmup_traces"]
        output.warmup_tokens = data['token_stats']["warmup_tokens"]
        output.entropy_bar = data.get("entropy_bar", None)
        return True
    