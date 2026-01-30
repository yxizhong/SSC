"""
Output classes for DeepThinkLLM

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class DeepThinkOutput:
    """Output container for deep thinking results"""
    
    # Primary results
    final_answer: Optional[str] = None
    voted_answer: Optional[str] = None  # Default voting method result
    
    # Multiple voting results
    voting_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Traces and voting
    warmup_traces: List[Dict[str, Any]] = field(default_factory=list)
    final_traces: List[Dict[str, Any]] = field(default_factory=list)
    all_traces: List[Dict[str, Any]] = field(default_factory=list)
    voting_answers: List[str] = field(default_factory=list)
    voting_weights: List[float] = field(default_factory=list)
    
    # Confidence information (for online mode)
    conf_bar: Optional[float] = None
    warmup_min_confs: List[float] = field(default_factory=list)
    final_min_confs: List[float] = field(default_factory=list)

    # token information
    entropy_bar: Optional[float] = None
    
    # Statistics
    total_traces_count: int = 0
    
    # Token statistics
    warmup_tokens: int = 0
    final_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_trace: float = 0.0
    avg_tokens_per_warmup_trace: float = 0.0
    avg_tokens_per_final_trace: float = 0.0
    
    # Timing information
    tokenizer_init_time: float = 0.0
    llm_init_time: float = 0.0
    warmup_gen_time: float = 0.0
    warmup_process_time: float = 0.0
    final_gen_time: float = 0.0
    final_process_time: float = 0.0
    generation_time: float = 0.0
    processing_time: float = 0.0
    total_time: float = 0.0
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    mode: str = "offline"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            # Primary results
            "final_answer": self.final_answer,
            "voted_answer": self.voted_answer,
            
            # Multiple voting results
            "voting_results": self.voting_results,
            
            # Traces and voting
            "warmup_traces": self.warmup_traces,
            "final_traces": self.final_traces,
            "all_traces": self.all_traces,
            "voting_answers": self.voting_answers,
            "voting_weights": self.voting_weights,
            
            # Confidence information
            "conf_bar": self.conf_bar,
            "warmup_min_confs": self.warmup_min_confs,
            "final_min_confs": self.final_min_confs,
            
            "entropy_bar": self.entropy_bar,
            
            # Statistics
            "total_traces_count": self.total_traces_count,
            
            # Token statistics
            "token_stats": {
                "warmup_tokens": self.warmup_tokens,
                "final_tokens": self.final_tokens,
                "total_tokens": self.total_tokens,
                "avg_tokens_per_trace": self.avg_tokens_per_trace,
                "avg_tokens_per_warmup_trace": self.avg_tokens_per_warmup_trace,
                "avg_tokens_per_final_trace": self.avg_tokens_per_final_trace,
            },
            
            # Timing information
            "timing_stats": {
                "tokenizer_init_time": self.tokenizer_init_time,
                "llm_init_time": self.llm_init_time,
                "warmup_gen_time": self.warmup_gen_time,
                "warmup_process_time": self.warmup_process_time,
                "final_gen_time": self.final_gen_time,
                "final_process_time": self.final_process_time,
                "generation_time": self.generation_time,
                "processing_time": self.processing_time,
                "total_time": self.total_time,
            },
            
            # Configuration and metadata
            "config": self.config,
            "mode": self.mode,
            "timestamp": self.timestamp,
        }
    
    def print_summary(self):
        """Print a formatted summary of the results"""
        print(f"\n=== Deep Thinking Summary ===")
        print(f"Mode: {self.mode}")
        
        if self.mode == "online":
            print(f"Warmup traces: {len(self.warmup_traces)}")
            print(f"Final traces: {len(self.final_traces)}")
            if self.conf_bar is not None:
                print(f"Confidence threshold: {self.conf_bar:.3f}")
        else:
            print(f"Generated traces: {self.total_traces_count}")
        
        print(f"Valid answers for voting: {len(self.voting_answers)}")
        
        if self.final_answer:
            print(f"Final answer: {self.final_answer}")
        
        print(f"Total tokens: {self.total_tokens}")
        
        if self.mode == "online":
            print(f"Warmup tokens: {self.warmup_tokens}, Final tokens: {self.final_tokens}")
            if self.warmup_gen_time > 0:
                print(f"Warmup time: {self.warmup_gen_time:.2f}s gen, {self.warmup_process_time:.2f}s proc")
            if self.final_gen_time > 0:
                print(f"Final time: {self.final_gen_time:.2f}s gen, {self.final_process_time:.2f}s proc")
        else:
            if self.generation_time > 0:
                print(f"Generation time: {self.generation_time:.2f}s")
                print(f"Generation throughput: {self.total_tokens / self.generation_time:.1f} tokens/second")
        
        print(f"Total time: {self.total_time:.2f}s")
        
        # Print voting results summary
        if self.voting_results:
            print(f"\n=== Voting Results Summary ===")
            for method, result in self.voting_results.items():
                if result and result.get('answer'):
                    confidence = result.get('confidence')
                    num_votes = result.get('num_votes', 0)
                    
                    conf_str = f" (conf: {confidence:.3f})" if confidence is not None else ""
                    
                    print(f"  {method}: {result['answer']}{conf_str} [{num_votes} votes]")
    
    def print_detailed_voting_results(self):
        """Print detailed voting results"""
        if not self.voting_results:
            print("No voting results available.")
            return
        
        print(f"\n=== Detailed Voting Results ===")
        print("-" * 70)
        print(f"{'Method':<25} {'Answer':<20} {'Votes':<6} {'Confidence':<12}")
        print("-" * 70)
        
        for method, result in self.voting_results.items():
            if result:
                answer = result.get('answer', 'None')[:18] + '...' if len(str(result.get('answer', 'None'))) > 20 else str(result.get('answer', 'None'))
                num_votes = result.get('num_votes', 0)
                confidence = result.get('confidence')
                
                conf_str = f"{confidence:.3f}" if confidence is not None else '-'
                
                print(f"{method:<25} {answer:<20} {num_votes:<6} {conf_str:<12}")
    
    @property
    def warmup_total_time(self) -> float:
        """Combined warmup generation and processing time"""
        return self.warmup_gen_time + self.warmup_process_time
    
    @property
    def final_total_time(self) -> float:
        """Combined final generation and processing time"""
        return self.final_gen_time + self.final_process_time
    
    @property
    def overall_throughput(self) -> float:
        """Overall token generation throughput"""
        total_gen_time = self.warmup_gen_time + self.final_gen_time + self.generation_time
        if total_gen_time > 0:
            return self.total_tokens / total_gen_time
        return 0.0
    
    def get_voting_method_names(self) -> List[str]:
        """Get list of available voting method names"""
        return list(self.voting_results.keys())
    
    def get_voting_answers(self) -> Dict[str, str]:
        """Get answers from all voting methods"""
        return {method: result.get('answer') for method, result in self.voting_results.items() 
                if result and result.get('answer')}