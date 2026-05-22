import re
import os
import torch

from vllm import LLM, SamplingParams
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


class PRMScorer:
    def __init__(self, model="Qwen/Qwen2.5-Math-PRM-7B", gpu_id=1, gpu_memory_utilization=0.9, tensor_parallel_size=1, max_model_len=4096, task="reward"):        
        self.llm = LLM(
            model=model,
            task=task,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=gpu_id,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )


    def score(self, text, context=""):
        full_text = context + text if context else text
        try:
            outputs = self.llm.generate([full_text], self.sampling_params)
            if outputs[0].outputs[0].logprobs:
                logprobs = [lp for token_logprobs in outputs[0].outputs[0].logprobs
                           for lp in token_logprobs.values()]
                score = sum(logprobs) / len(logprobs) if logprobs else 0.0
            else:
                score = 0.0
            return float(score)
        except:
            return 0.0

    def score_batch(self, texts, contexts=None):
        if contexts is None:
            contexts = [""] * len(texts)
        full_texts = [c + t for c, t in zip(contexts, texts)]

        try:
            outputs = self.llm.generate(full_texts, self.sampling_params)
            scores = []
            for output in outputs:
                if output.outputs[0].logprobs:
                    logprobs = [lp for token_logprobs in output.outputs[0].logprobs
                               for lp in token_logprobs.values()]
                    score = sum(logprobs) / len(logprobs) if logprobs else 0.0
                else:
                    score = 0.0
                scores.append(score)
            return torch.tensor(scores)
        except:
            return torch.zeros(len(texts))