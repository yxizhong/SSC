import os
import time
import copy
import numpy as np
from tqdm import tqdm
from math import ceil
from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams
from .wrapper import DeepThinkLLM
from .utils import process_batch_results_bystep
from .outputs import DeepThinkOutput
from .selfstepconf_processors import WrappedPerReqLogitsProcessor

class SelfStepConfDeepThinkLLM(DeepThinkLLM):
    def __init__(self, model: str, **vllm_kwargs):
        # Call parent class initialization
        super().__init__(model, processors=WrappedPerReqLogitsProcessor, **vllm_kwargs)
        self.step_tokens = ["\n\n"]
        self.reflection_token = "wait"
    
    def _deepthink_selfstepconf(
        self,
        prompt: str,
        output: DeepThinkOutput,
        total_budget,
        sampling_params: Optional[SamplingParams],
        save_path: str,
        extra_args=None
    ) -> DeepThinkOutput:
    
        processing_start = time.time()

        # Final phase
        print(f"Starting final phase...", sampling_params)
        final_gen_start = time.time()
        
        base_seed = time.time_ns()
        base_seed = 20251103
        final_params_list = []
        for param_id in range(total_budget):
            final_params = copy.deepcopy(sampling_params) 
            final_params.logprobs = 20
            final_params.seed = base_seed + param_id
            if total_budget == 1:
                final_save_path = save_path
            else:
                final_save_path = save_path.split('.')[0] + f"_{param_id+1}" + ".pkl"
            final_params.extra_args = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "step_token_ids": [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens],
                "reflection_token_ids": self.tokenizer.encode(self.reflection_token, add_special_tokens=False)[0],
                "conf_topk": 20,
                "model_name": self.model_name,
                "save_path": final_save_path,
                "extra_args": extra_args,
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