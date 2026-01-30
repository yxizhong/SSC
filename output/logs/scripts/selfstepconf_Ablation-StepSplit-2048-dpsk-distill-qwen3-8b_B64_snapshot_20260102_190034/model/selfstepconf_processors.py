import time
import pickle
import torch
import functools
import multiprocessing
import numpy as np

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

    def __init__(self, eos_token_id: int, step_token_ids: list[int], conf_topk: int, reflection_token_ids: int, model_name: str, save_path: str, extra_args=None) -> None:
        """Specify `confidence`"""
        self.beta = extra_args.get("beta", 0.95)
        self.alpha = extra_args.get("alpha", 0.9)
        self.delta = extra_args.get("delta", 0.8)
        self.conf_bar = extra_args.get("conf_bar", 0.0)
        print(f"beta: {self.beta}, alpha: {self.alpha}, delta: {self.delta}, conf_bar: {self.conf_bar}")

        self.threshold = self.conf_bar
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

        # self.delta = max(1-(self.token_idx_count/1000)*(1-0.8), 0.7)

        # if self.step_token in output_token:
        # if output_ids in self.step_token_ids:
        if self.token_idx_count%2048==0 and output_ids != self.eos_token_id:
            current_step_confs = self.step_conf_list["step_confs"][-1]
            step_conf = sum(current_step_confs)/len(current_step_confs)
            self.step_conf_list["confs"].append(step_conf)

            upward = True
            if len(self.step_conf_list["confs"]) > 1:
                upward = (self.step_conf_list["confs"][-1] > self.step_conf_list["confs"][-2])
                # previous_var = np.var(self.step_conf_list["confs"][:-1])
                # current_var = np.var(self.step_conf_list["confs"])
                # if previous_var !=0 and current_var > previous_var:
                #     upward = False

            if self.threshold > 0 and step_conf/self.threshold < self.delta and not upward:
            # if self.threshold > 0 and not upward:
                # update prompt "wait, [feedback]"
                # self.threshold = self.alpha*self.threshold + (1-self.alpha)*step_conf
                return self.reflection(logits, stop_reason="stepconf_below_threshold")
            else:
                if self.threshold > 0:
                    # current_step_length = [len(step) for step in self.step_conf_list["step_token_ids"]]
                    # self.threshold = sum([self.step_conf_list["confs"][idx]*current_step_length[idx] for idx in range(len(current_step_length))])/sum(current_step_length)
                    self.threshold = self.alpha*self.threshold + (1-self.alpha)*step_conf
                else:
                    self.threshold = step_conf
                
                self.step_conf_list["step_start_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_end_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_token_ids"].append([])
                self.step_conf_list["step_confs"].append([])
                self.step_conf_list["stop_reason"].append("stop")
                self.step_conf_list["threshold"].append(self.threshold)

                # print(f"update threshold to {self.threshold}")

        elif output_ids == self.eos_token_id:
            final_step_confs = self.step_conf_list["step_confs"][-1]

            final_step_token_ids = self.step_conf_list["step_token_ids"][-1]
            final_output = self.tokenizer.decode(final_step_token_ids)
            answer = extract_answer(final_output)
            
            final_conf = sum(final_step_confs)/len(final_step_confs)
            self.step_conf_list["confs"].append(final_conf)
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
                # previous_var = np.var(self.step_conf_list["confs"][:-1])
                # current_var = np.var(self.step_conf_list["confs"])
                # if previous_var !=0 and current_var > previous_var:
                #     upward = False

            if self.threshold > 0 and final_conf/self.threshold < self.delta and not stop and not upward:
            # if self.threshold > 0 and not upward:
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
    
    def adaptive_stop(self):
        if self.answer_count:
            majority_answer = max(self.answer_count, key=self.answer_count.get)
            beta = self.answer_weight[majority_answer]/sum(self.answer_weight.values())
        
            if beta >= self.beta and sum(self.answer_count.values())>=2:
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


class HETConfPerReqLogitsProcessor:
    """
    High-Entropy-Token split step LogitsProcessor
    """

    def __init__(self, eos_token_id: int, step_token_ids: list[int], conf_topk: int, reflection_token_ids: int, model_name: str, save_path: str, extra_args=None) -> None:
        """Specify `confidence`"""
        self.threshold = 0.0
        self.HET_threshold = 0.672
        self.entropy_percentile = 20
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
                "step_entropys": [[]],
                "stop_reason": ["stop"],
                "threshold": [self.threshold],
                "HET_threshold": [self.HET_threshold]
        }
        self.answer_count = {}
        self.answer_weight = {}
        self.token_idx_count = 0
        self.beta = extra_args.get("beta", 0.95)
        self.alpha = extra_args.get("alpha", 0.9)
        self.delta = extra_args.get("delta", 0.8)
        print(f"beta: {self.beta}, alpha: {self.alpha}, delta: {self.delta}, HET_threshold: {self.HET_threshold}")
        self.cd_num = 0
        self.step_signal = False
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
            if self.threshold > 0 and final_conf/self.threshold < self.delta and not stop:
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

        # elif new_entropy > self.HET_threshold and len(self.step_conf_list["step_confs"][-1]) >= 200:
        elif output_ids in self.step_token_ids:
            current_step_confs = self.step_conf_list["step_confs"][-1]
            step_conf = sum(current_step_confs)/len(current_step_confs)
            if self.threshold > 0 and step_conf/self.threshold < self.delta:
                # update prompt "wait, [feedback]"
                return self.reflection(logits, stop_reason="stepconf_below_threshold")
            else:
                if self.threshold > 0:
                    self.threshold = self.alpha*self.threshold + (1-self.alpha)*step_conf
                else:
                    self.threshold = step_conf
                
                # update HET_threshold
                step_entropys = self.step_conf_list["step_entropys"][-1]
                step_entropy = np.percentile(step_entropys, 100 - self.entropy_percentile)
                self.HET_threshold = self.alpha*self.HET_threshold + (1-self.alpha)*step_entropy
                
                self.step_conf_list["step_start_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_end_idx"].append(self.token_idx_count+1)
                self.step_conf_list["step_token_ids"].append([])
                self.step_conf_list["step_confs"].append([])
                self.step_conf_list["step_entropys"].append([])
                self.step_conf_list["stop_reason"].append("stop")
                self.step_conf_list["threshold"].append(self.threshold)
                self.step_conf_list["HET_threshold"].append(self.HET_threshold)

                # print(f"update threshold to {self.threshold}")

        if self.cd_num != 0 :
            self.cd_num -= 1
        return logits
    
    def adaptive_stop(self):
        if self.answer_count:
            majority_answer = max(self.answer_count, key=self.answer_count.get)
            beta = self.answer_weight[majority_answer]/sum(self.answer_weight.values())
        
            if beta >= self.beta and sum(self.answer_count.values())>=2:
                stop = True
                print(f"early stop, beta={beta}")
            else:
                stop = False
        else:
            stop = False
        
        return stop
    
    def reflection(self, logits, stop_reason):
        if not self.wait_token_ids:
            self.cd_num = 5
            self.step_conf_list["stop_reason"][-1] = stop_reason

            wait_tokens = []
            max_entropy_tokens = ""
            max_entropy = 0.0
            self.wait_token_ids = [self.reflection_token_ids]
            for idx in range(len(self.step_conf_list["step_entropys"][-1])):
                curr_token_entropy = self.step_conf_list["step_entropys"][-1][idx]
                if curr_token_entropy >= self.HET_threshold:
                    if curr_token_entropy >= max_entropy:
                        max_entropy = curr_token_entropy
                        max_entropy_tokens = self.tokenizer.decode(self.step_conf_list["step_token_ids"][-1][idx])
                    # wait_tokens.append(self.tokenizer.decode(self.step_conf_list["step_token_ids"][-1][idx]))
            # wait_tokens = [wait_token for wait_token in wait_tokens if len(wait_token.strip()) > 1]
            
            if max_entropy_tokens:
                reflects = f", the information after <{max_entropy_tokens}> is important"
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
            print("Not using ConfPerReqLogitsProcessor", params.extra_args)
            return None
        print(f"Using ConfPerReqLogitsProcessor with eos_token_id {eos_token_id}, step_token_ids {step_token_ids}, topk {conf_topk}, reflection_token_ids {reflection_token_ids}, model_name {model_name}, save_path {save_path}")
        return ConfPerReqLogitsProcessor(eos_token_id, step_token_ids, conf_topk, reflection_token_ids, model_name, save_path, params.extra_args.get("extra_args", None))
        # return HETConfPerReqLogitsProcessor(eos_token_id, step_token_ids, conf_topk, reflection_token_ids, model_name, save_path, params.extra_args.get("extra_args", None))