<div align="center">

# Beyond High-Entropy Exploration: Correctness-Aware Low-Entropy Segment-Based Advantage Shaping for Reasoning LLMs.


# Getting started

After preparing the training data, for training Qwen2.5-7B on a single node, you can simply run:

```
cd verl
conda activate your_env
bash 7b_base.sh
```

# Acknowledgement
We implement our reinforcement learning algorithm extending from [verl](https://github.com/volcengine/verl). We utilize [vLLM](https://github.com/vllm-project/vllm) for inference. Our models are trained primarily on [Qwen2.5 family](https://github.com/QwenLM/Qwen2.5). Our training data is built from [hendrycks_math](https://huggingface.co/datasets/hendrydong/hendrycks_math). Thanks for their great contributions!


