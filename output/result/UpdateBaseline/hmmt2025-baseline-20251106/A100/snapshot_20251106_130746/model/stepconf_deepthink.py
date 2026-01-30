import os
import time
import copy
import glob
import numpy as np
from tqdm import tqdm
from math import ceil
from datetime import datetime
from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams
from .wrapper import DeepThinkLLM
from .utils import process_batch_results_bystep, compute_least_step
from .outputs import DeepThinkOutput
from .stepconf_processors import WrappedPerReqLogitsProcessor

class StepConfDeepThinkLLM(DeepThinkLLM):
    def __init__(self, model: str, **vllm_kwargs):
        # Call parent class initialization
        super().__init__(model, processors=WrappedPerReqLogitsProcessor, **vllm_kwargs)
        self.step_tokens = ["\n\n", ]
        self.reflection_token = "wait"
    
    def _deepthink_stepconf(
        self,
        prompt: str,
        output: DeepThinkOutput,
        warmup_traces: int,
        confidence_percentile: int,
        sampling_params: Optional[SamplingParams],
        save_path: str,
        warmup_traces_dir: str
    ) -> DeepThinkOutput:
    
        processing_start = time.time()
        # Warmup phase
        print(f"Starting warmup phase...", sampling_params)
        warmup_gen_start = time.time()
        # Generate warmup traces
        warmup_params_list = []
        base_seed = time.time_ns()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tokenentropy_save_path = "/".join(save_path.split('/')[:-1]) + f"/tokenentropy_{timestamp}_" + save_path.split('/')[-1]
        if not self.load_warmup_traces(output, save_path, warmup_traces_dir):
            for param_id in range(warmup_traces):
                warmup_params = copy.deepcopy(sampling_params) 
                warmup_params.logprobs = 20
                warmup_params.seed = base_seed + param_id
                warmup_params.extra_args = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "save_path": tokenentropy_save_path
                }
                warmup_params_list.append(warmup_params)
            warmup_outputs = self.llm.generate([prompt for _ in range(warmup_traces)], warmup_params_list)
            output.warmup_gen_time = time.time() - warmup_gen_start

            # Process warmup results
            warmup_process_start = time.time()
            # warmup_result = process_batch_results(warmup_outputs, 2048)
            warmup_result = process_batch_results_bystep(warmup_outputs, [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens])
            output.warmup_process_time = time.time() - warmup_process_start

            print('Warmup min_confs:', warmup_result['min_confs'])
            output.conf_bar = float(np.percentile(warmup_result['min_confs'],100 - confidence_percentile))
            output.warmup_min_confs = warmup_result['min_confs']
            
            output.warmup_traces = warmup_result['traces']
            output.warmup_tokens = warmup_result['total_tokens']
            output.entropy_bar = float(self.computes_entropy_bar(tokenentropy_save_path))
            print(f"Warmup completed: conf_bar={output.conf_bar:.3f}, entropy_bar={output.entropy_bar:.3f}")
        else:
            print("load warmup traces successfully!")
            output.conf_bar = float(np.percentile(output.warmup_min_confs, 100 - confidence_percentile))
        
        # output.entropy_bar = float(np.percentile(np.concatenate([trace["confs"] for trace in output.warmup_traces]), 20))
        # tokenentropy_save_path = glob.glob(os.path.join(warmup_traces_dir, "tokenentropy_*" + save_path.split('/')[-1]))[0]
        # output.entropy_bar = float(self.computes_entropy_bar(tokenentropy_save_path, entropy_percentile=5))

        # Final phase
        print(f"Starting final phase...", sampling_params)
        final_gen_start = time.time()
        
        final_params_list = []
        conf_bar_list = list(set(float(np.percentile(output.warmup_min_confs[:idx+1], 100 - confidence_percentile)) for idx in range(len(output.warmup_min_confs))))
        for param_id, conf_bar in enumerate(conf_bar_list):
            final_params = copy.deepcopy(sampling_params) 
            final_params.logprobs = 20
            final_params.seed = base_seed + warmup_traces + param_id
            
            if len(output.warmup_min_confs) == 1:
                final_save_path = save_path
            else:
                final_save_path = save_path.split('.')[0] + f"_{param_id+1}" + ".pkl"
            final_params.extra_args = {
                "conf_threshold": conf_bar,
                "eos_token_id": self.tokenizer.eos_token_id,
                "step_token_ids": [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens],
                "HET_threshold": output.entropy_bar,
                "reflection_token_ids": self.tokenizer.encode(self.reflection_token, add_special_tokens=False)[0],
                "conf_topk": 20,
                "model_name": self.model_name,
                "save_path": final_save_path,
            }
            final_params_list.append(final_params)
        final_outputs = self.llm.generate([prompt for _ in range(len(conf_bar_list))], final_params_list)
        output.final_gen_time = time.time() - final_gen_start
        
        # Process final results
        final_process_start = time.time()
        final_result = process_batch_results_bystep(final_outputs, [self.tokenizer.encode(step_token, add_special_tokens=False)[0] for step_token in self.step_tokens])
        output.final_process_time = time.time() - final_process_start
        
        for idx, trace in enumerate(final_result["traces"]):
            trace["conf_bar"] = conf_bar_list[idx]

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
    
    def computes_entropy_bar(self, tokenentropy_save_path, entropy_percentile=20):
        try:
            # Try torch.load first for CUDA tensors, fallback to pickle
            try:
                import torch
                import pickle
                data = torch.load(tokenentropy_save_path, map_location='cpu')
            except:
                with open(tokenentropy_save_path, "rb") as f:
                    data = pickle.load(f)

        except Exception as e:
            print(f"Error loading {tokenentropy_save_path}: {e}")
            return None
        
        if type(data[0]) == dict:
            token_entropys = np.concatenate([item["token_entropy"] for item in data])
        elif type(data[0]) == list:
            token_entropys = np.concatenate([item for item in data])
        entropy_bar = np.percentile(token_entropys, 100-entropy_percentile)

        return entropy_bar
    
    def load_warmup_traces(self, output, save_path, warmup_traces_dir):
        qid = save_path.split('_')[-1].split('.')[0]
        pattern_path = os.path.join(warmup_traces_dir, f"deepthink_online_baseline_qid{qid}*.pkl")
        file_paths = glob.glob(pattern_path)
        
        if len(file_paths) != 0:
            file_path = file_paths[0]
        else:
            return False

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