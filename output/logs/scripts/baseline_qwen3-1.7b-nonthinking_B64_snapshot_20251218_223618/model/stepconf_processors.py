import os
import time
import json
import torch
import pickle
import functools
import multiprocessing

from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import VllmConfig
from collections import deque
from typing import Optional, List, Callable, Any, Dict
from abc import ABC, abstractmethod
from .utils import extract_answer

class ConfPerReqLogitsProcessor:
    """The request-level logits processor masks out all logits except the
    token id identified by `target_token`"""

    def __init__(self, threshold: float, eos_token_id: int, step_token_ids: list[int], conf_topk: int, reflection_token_ids: int, model_name: str, save_path) -> None:
        """Specify `confidence`"""
        self.threshold = threshold
        self.eos_token_id = eos_token_id
        self.step_token_ids = step_token_ids
        self.conf_topk = conf_topk
        self.reflection_token_ids = reflection_token_ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.save_path = save_path
        self.step_conf_list = {
                "step_start_idx": [0],
                "step_end_idx": [0],
                "step_token_ids": [[]],
                "step_confs": [[]],
                "confs": [],
                "stop_reason": ["stop"],
                "threshold": [self.threshold]
        }
        self.answer_count = {}
        self.answer_weight = {}
        self.token_idx_count = 0
        self.step_token = self.tokenizer.decode(self.step_token_ids)
        self.need_reflection = False

    def compute_conf(self, logits: torch.Tensor) -> float:
        # Compute the confidence score based on the logits
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probabilities, self.conf_topk, dim=-1)
        log_probs = torch.log(top_probs)
        return -log_probs.sum().item() / self.conf_topk

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        
        self.token_idx_count += 1
        new_conf = self.compute_conf(logits)
        output_ids = torch.argmax(logits).cpu().numpy()
        output_token = self.tokenizer.decode(output_ids) 

        self.step_conf_list["step_end_idx"][-1] = self.token_idx_count
        self.step_conf_list["step_confs"][-1].append(new_conf)
        self.step_conf_list["step_token_ids"][-1].append(output_ids)

        if self.need_reflection:
            return self.reflection(logits, stop_reason=None)

        if output_ids in self.step_token_ids:
        # if self.step_token in output_token:
            current_step_confs = self.step_conf_list["step_confs"][-1]
            step_conf = sum(current_step_confs)/len(current_step_confs)
            self.step_conf_list["confs"].append(step_conf)

            upward = True
            if len(self.step_conf_list["confs"]) > 1:
                upward = (self.step_conf_list["confs"][-1] > self.step_conf_list["confs"][-2])

            # if sum(current_step_confs)/len(current_step_confs) < self.threshold:
            if step_conf < self.threshold and not upward:
                # update prompt "wait, [feedback]"
                return self.reflection(logits, stop_reason="stepconf_below_threshold")
            else:
                self.step_conf_list["step_start_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_end_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_token_ids"].append([])
                self.step_conf_list["step_confs"].append([])
                self.step_conf_list["stop_reason"].append("stop")
                self.step_conf_list["threshold"].append(self.threshold)

        elif output_ids == self.eos_token_id:
            final_step_confs = self.step_conf_list["step_confs"][-1]

            final_step_token_ids = self.step_conf_list["step_token_ids"][-1]
            final_output = self.tokenizer.decode(final_step_token_ids)
            answer = extract_answer(final_output)
            
            final_conf = sum(final_step_confs)/len(final_step_confs)
            if not answer:
                return self.reflection(logits, stop_reason="extracted_answer_failed")
            else:
                if answer not in self.answer_count:
                    self.answer_count[answer] = 1
                    self.answer_weight[answer] = final_conf
                else:
                    self.answer_count[answer] += 1
                    self.answer_weight[answer] += final_conf

            stop = self.adaptive_stop()
            upward = True
            if len(self.step_conf_list["confs"]) > 1:
                upward = (self.step_conf_list["confs"][-1] > self.step_conf_list["confs"][-2])

            # if final_conf < self.threshold and not stop:
            if final_conf < self.threshold and not stop and not upward:
                return self.reflection(logits, stop_reason="answerconf_below_threshold")
            else:
                res = {
                    "step": self.step_conf_list,
                    "answer_count": self.answer_count,
                    "answer_weight": self.answer_weight
                }

                with open(self.save_path, 'wb') as pkl_file:
                    pickle.dump(res, pkl_file)
                print(f"res save to {self.save_path}")
                return logits
        
        if self.token_idx_count == 64000:
            res = {
                "step": self.step_conf_list,
                "answer_count": self.answer_count,
                "answer_weight": self.answer_weight
            }

            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(res, pkl_file)
            print(f"length limit, partial res save to {self.save_path}")
        
        return logits
    
    def adaptive_stop(self, threshold=0.95):
        if self.answer_count:
            majority_answer = max(self.answer_count, key=self.answer_count.get)
            beta = self.answer_weight[majority_answer]/sum(self.answer_weight.values())
        
            if beta >= threshold and sum(self.answer_count.values())>=2:
                stop = True
                print(f"early stop, beta={beta}")
            else:
                stop = False
        else:
            stop = False
        
        return stop
    
    def reflection(self, logits, stop_reason):
        if not self.need_reflection:
            self.step_conf_list["stop_reason"][-1] = stop_reason
            self.step_conf_list["step_start_idx"].append(self.token_idx_count+1)
            self.step_conf_list["step_end_idx"].append(self.token_idx_count+1)
            self.step_conf_list["step_token_ids"].append([])
            self.step_conf_list["step_confs"].append([])
            self.step_conf_list["stop_reason"].append("stop")
            self.step_conf_list["threshold"].append(self.threshold)
            self.need_reflection = True
        
        # update prompt "wait, the correct answer output should be in the form of "\boxed{[answer]}"."
        # keep_token_id = self.reflection_token_ids
        # val_to_keep = logits[keep_token_id].item()
        # logits[:] = float("-inf")
        # logits[keep_token_id] = val_to_keep
        else:
            keep_token_id = self.reflection_token_ids
            val_to_keep = logits[keep_token_id].item()
            replace_token_id = logits.argmax()
            val_to_replace = logits[replace_token_id].item()
            logits[keep_token_id] = val_to_replace
            logits[replace_token_id] = val_to_keep
            self.need_reflection = False

            # self.step_conf_list["step_confs"][-1][-1] = -1
            self.step_conf_list["step_token_ids"][-1][-1] = self.reflection_token_ids
        
        if self.token_idx_count == 64000:
            res = {
                "step": self.step_conf_list,
                "answer_count": self.answer_count,
                "answer_weight": self.answer_weight
            }

            with open(self.save_path, 'wb') as pkl_file:
                pickle.dump(res, pkl_file)
            print(f"length limit, partial res save to {self.save_path}")
        
        return logits

class TokenEntropyLogitsProcessor:
    def __init__(self, eos_token_id, save_path):
        self.token_entropy = []
        self.output_ids = []
        self.eos_token_id = eos_token_id
        self.save_path = save_path
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        # Compute the confidence score based on the logits
        probabilities = torch.softmax(logits, dim=-1)
        new_entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
        
        return new_entropy.cpu().numpy()

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        output_ids = torch.argmax(logits).cpu().numpy()
        self.output_ids.append(output_ids)
        self.token_entropy.append(self.compute_entropy(logits))
        
        if output_ids == self.eos_token_id:
            res = {
                "output_ids": self.output_ids,
                "token_entropy": self.token_entropy
            }

            lock_path = self.save_path + ".lock"
            # lock
            while os.path.exists(lock_path):
                print(f"⏳ {lock_path} exists, waiting...")
                time.sleep(0.5)

            open(lock_path, 'w').close()

            try:
                if os.path.exists(self.save_path):
                    with open(self.save_path, 'rb') as pkl_file:
                        try:
                            old_res = pickle.load(pkl_file)
                        except EOFError:
                            old_res = []
                    old_res.append(res)
                    with open(self.save_path, 'wb') as pkl_file:
                        pickle.dump(old_res, pkl_file)
                    print(f"save token entropy at {self.save_path}, len {len(old_res)}")
                else:
                    with open(self.save_path, 'wb') as pkl_file:
                        pickle.dump([res], pkl_file)
                    print(f"save token entropy at {self.save_path}, len 1")
            finally:
                os.remove(lock_path)
        
        return logits


class HETConfPerReqLogitsProcessor:
    """
    High-Entropy-Token 
    """

    def __init__(self, threshold: float, HET_threshold: float, step_token_ids: list[int], eos_token_id: int, conf_topk: int, reflection_token_ids: int, model_name: str, save_path) -> None:
        """Specify `confidence`"""
        self.threshold = threshold
        self.HET_threshold = HET_threshold
        self.step_token_ids = step_token_ids
        self.eos_token_id = eos_token_id
        self.conf_topk = conf_topk
        self.reflection_token_ids = reflection_token_ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.save_path = save_path
        self.step_conf_list = {
                "step_start_idx": [0],
                "step_end_idx": [0],
                "step_token_ids": [[]],
                "step_confs": [[]],
                "step_entropys": [[]],
                "stop_reason": ["stop"],
                "threshold": [self.threshold],
                "HET_threshold": [self.HET_threshold]
        }
        self.answer_count = {}
        self.answer_weight = {}
        self.token_idx_count = 0
        self.cd_num = 0
        self.step_singal = False
        self.wait_token_ids = []

    def compute_conf(self, logits: torch.Tensor) -> float:
        # Compute the confidence score based on the logits
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probabilities, self.conf_topk, dim=-1)
        log_probs = torch.log(top_probs)

        new_conf = -log_probs.sum().item() / self.conf_topk
        new_entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
        return new_conf, new_entropy.cpu().numpy()

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        
        self.token_idx_count += 1
        new_conf, new_entropy = self.compute_conf(logits)
        output_ids = torch.argmax(logits).cpu().numpy()

        self.step_conf_list["step_end_idx"][-1] = self.token_idx_count
        self.step_conf_list["step_confs"][-1].append(new_conf)
        self.step_conf_list["step_entropys"][-1].append(new_entropy)
        self.step_conf_list["step_token_ids"][-1].append(output_ids)

        if self.wait_token_ids:
            return self.reflection(logits, stop_reason="")

        if output_ids == self.eos_token_id:
            final_step_confs = self.step_conf_list["step_confs"][-1]

            final_step_token_ids = self.step_conf_list["step_token_ids"][-1]
            final_output = self.tokenizer.decode(final_step_token_ids)
            answer = extract_answer(final_output)
            
            final_conf = sum(final_step_confs)/len(final_step_confs)
            if not answer:
                return self.reflection(logits, stop_reason="extracted_answer_failed")
            else:
                if answer not in self.answer_count:
                    self.answer_count[answer] = 1
                    self.answer_weight[answer] = final_conf
                else:
                    self.answer_count[answer] += 1
                    self.answer_weight[answer] += final_conf

            stop = self.adaptive_stop()
            if final_conf < self.threshold and not stop:
                return self.reflection(logits, stop_reason="answerconf_below_threshold")
            else:
                res = {
                    "step": self.step_conf_list,
                    "answer_count": self.answer_count,
                    "answer_weight": self.answer_weight
                }
                
                with open(self.save_path, 'wb') as pkl_file:
                    pickle.dump(res, pkl_file)
                print(f"res save to {self.save_path}")

        # elif new_conf < self.HET_threshold and len(self.step_conf_list["step_confs"][-1]) >= 200:
        # elif new_entropy > self.HET_threshold and len(self.step_conf_list["step_confs"][-1]) >= 200:
        elif new_entropy > self.HET_threshold:
        # elif output_ids in self.step_token_ids:
            current_step_confs = self.step_conf_list["step_confs"][-1]

            if sum(current_step_confs)/len(current_step_confs) < self.threshold:
                # update prompt "wait, [feedback]"
                return self.reflection(logits, stop_reason="stepconf_below_threshold")
            else:
                self.step_conf_list["step_start_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_end_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_token_ids"].append([])
                self.step_conf_list["step_confs"].append([])
                self.step_conf_list["step_entropys"].append([])
                self.step_conf_list["stop_reason"].append("stop")
                self.step_conf_list["threshold"].append(self.threshold)
                self.step_conf_list["HET_threshold"].append(self.HET_threshold)
        
        if self.cd_num != 0:
            self.cd_num -= 1
        
        return logits
    
    def adaptive_stop(self, threshold=0.95):
        if self.answer_count:
            majority_answer = max(self.answer_count, key=self.answer_count.get)
            beta = self.answer_weight[majority_answer]/sum(self.answer_weight.values())
        
            if beta >= threshold and sum(self.answer_count.values())>=2:
                stop = True
                print(f"early stop, beta={beta}")
            else:
                stop = False
        else:
            stop = False
        
        return stop
    
    def reflection(self, logits, stop_reason):
        return self._reflection(logits, stop_reason)
        if not self.wait_token_ids:
            self.cd_num = 5
            self.step_conf_list["stop_reason"][-1] = stop_reason

            wait_tokens = []
            if len(self.step_conf_list["step_entropys"][-1]) != len(self.step_conf_list["step_token_ids"][-1]):
                print("Error")
            for idx in range(len(self.step_conf_list["step_entropys"][-1])):
                if self.step_conf_list["step_entropys"][-1][idx] >= self.HET_threshold:
                    self.wait_token_ids = [self.reflection_token_ids]
                    wait_tokens.append(self.tokenizer.decode(self.step_conf_list["step_token_ids"][-1][idx]))
            wait_tokens = [wait_token for wait_token in wait_tokens if len(wait_token.strip()) > 1]
            if wait_tokens:
                reflects = ", the information after [" + "],[".join(wait_tokens) + "] is important"
                self.wait_token_ids.extend(self.tokenizer.encode(reflects)[1:])
            else:
                self.wait_token_ids = [self.reflection_token_ids]
            
            self.step_conf_list["step_start_idx"].append(self.token_idx_count+1)
            self.step_conf_list["step_end_idx"].append(self.token_idx_count+1)
            self.step_conf_list["step_entropys"].append([])
            self.step_conf_list["step_token_ids"].append([])
            self.step_conf_list["step_confs"].append([])
            self.step_conf_list["stop_reason"].append("stop")
            self.step_conf_list["threshold"].append(self.threshold)
            self.step_conf_list["HET_threshold"].append(self.HET_threshold)
        
        # update prompt "wait, the correct answer output should be in the form of "\boxed{[answer]}"."
        keep_token_id = self.wait_token_ids.pop(0)
        val_to_keep = logits[keep_token_id].item()
        logits[:] = float("-inf")
        logits[keep_token_id] = val_to_keep
        
        return logits
    
    def _reflection(self, logits, stop_reason):
        self.step_conf_list["stop_reason"][-1] = stop_reason
        
        self.step_conf_list["step_start_idx"].append(self.token_idx_count+1)
        self.step_conf_list["step_end_idx"].append(self.token_idx_count+1)
        self.step_conf_list["step_entropys"].append([])
        self.step_conf_list["step_token_ids"].append([])
        self.step_conf_list["step_confs"].append([])
        self.step_conf_list["stop_reason"].append("stop")
        self.step_conf_list["threshold"].append(self.threshold)
        self.step_conf_list["HET_threshold"].append(self.HET_threshold)
        
        # update prompt "wait, the correct answer output should be in the form of "\boxed{[answer]}"."
        keep_token_id = self.reflection_token_ids
        val_to_keep = logits[keep_token_id].item()
        logits[:] = float("-inf")
        logits[keep_token_id] = val_to_keep
        
        return logits

class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Example of overriding the wrapper class `__init__()` in order to utilize
    info about the device type"""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self.is_cuda = device.type == "cuda"

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """This method returns a new request-level logits processor, customized
        to the `target_token` value associated with a particular request.

        Returns None if the logits processor should not be applied to the
        particular request. To use the logits processor the request must have
        a "target_token" custom argument with an integer value, and the device
        must be "cuda"-type

        Args:
          params: per-request sampling params

        Returns:
          `Callable` request logits processor, or None
        """
        if (
            not self.is_cuda
            or (
                conf_threshold := params.extra_args
                and params.extra_args.get("conf_threshold")
            )
            is None
            or (eos_token_id := params.extra_args
                and params.extra_args.get("eos_token_id")
            ) is None
            or (
                step_token_ids := params.extra_args
                and params.extra_args.get("step_token_ids")
            ) is None
            or (
                conf_topk := params.extra_args
                and params.extra_args.get("conf_topk")
            ) is None
            or (
                reflection_token_ids := params.extra_args
                and params.extra_args.get("reflection_token_ids")
            ) is None
            or (
                model_name := params.extra_args
                and params.extra_args.get("model_name")
            ) is None
            or (
                save_path := params.extra_args
                and params.extra_args.get("save_path")
            ) is None
        ):
            if (
                (eos_token_id := params.extra_args
                and params.extra_args.get("eos_token_id")
            ) is None 
            or (
                save_path := params.extra_args
                and params.extra_args.get("save_path")
            ) is None
            ):
                print("Not using ConfPerReqLogitsProcessor", params.extra_args)
                return None
            else:
                print(f"Using TokenEntropyLogitsProcessor with eos_token_id {eos_token_id}, save_path {save_path}")
                return TokenEntropyLogitsProcessor(eos_token_id, save_path)
        
        if HET_threshold := params.extra_args and params.extra_args.get("HET_threshold") is None:
            print(f"Using ConfPerReqLogitsProcessor with threshold {conf_threshold}, eos_token_id {eos_token_id}, step_token_ids {step_token_ids}, topk {conf_topk}, reflection_token_ids {reflection_token_ids}, model_name {model_name}, save_path {save_path}")
            return ConfPerReqLogitsProcessor(conf_threshold, eos_token_id, step_token_ids, conf_topk, reflection_token_ids, model_name, save_path)
        else:
            HET_threshold = params.extra_args.get("HET_threshold")
            print(f"Using HETConfPerReqLogitsProcessor with threshold {conf_threshold}, HET_threshold {HET_threshold}, eos_token_id {eos_token_id}, step_token_ids {step_token_ids}, topk {conf_topk}, reflection_token_ids {reflection_token_ids}, model_name {model_name}, save_path {save_path}")
            return HETConfPerReqLogitsProcessor(conf_threshold, HET_threshold, step_token_ids, eos_token_id, conf_topk, reflection_token_ids, model_name, save_path)

