<div align="center">

# The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning.


# Getting started

After preparing the training data, for training Qwen2.5-7B on a single node, you can simply run:

```
cd verl
conda activate your_env
bash 7b_base.sh
```

# 📖Introduction

<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/e2a.jpg?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

This paper addresses the entropy collapse issue in scaling reinforcement learning (RL) for large language models (LLMs), where policy entropy drops sharply during training, leading to overconfidence and performance saturation. We empirically establish a relationship between entropy ($H$) and performance ($R$): $R=−aexp(H)+b$, showing performance is bottlenecked by entropy exhaustion. 

<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/cov.jpg?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>

Theoretically, we find entropy changes are driven by the covariance between action probability and logit updates, which correlates with advantage in Policy Gradient methods. High-probability, high-advantage actions reduce entropy, while rare, high-advantage actions increase it. Empirically, the covariance term remains positive, explaining entropy’s monotonic decline. To mitigate this, we propose ​​Clip-Cov​​ and ​​KL-Cov​​, which restrict updates for high-covariance tokens. These methods effectively prevent entropy collapse, and improve performance. 

# 📃Evaluation

<div align="left">
  <img src="https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/figures/performance_fig.jpg?raw=true" alt="issue" style="width: 96%; height: auto;">
</div>


Our method is able to maintain a considerably higher level of entropy throughout training. For example, when the baseline's entropy reaches a plateau and can no longer be consumed, the KL-Cov method still sustains an entropy level over 10 times higher. Meanwhile, the response length of the policy model steadily increases, and its performance on the test set consistently surpasses that of the baseline. This indicates that our model is able to explore more freely during training, learning better policy through RL. 
| **Method**        | **AIME24** | **AIME25** |  **AMC** | **MATH-500** | **OMNI-MATH** | **OlympiadBench** | **Minerva** | **Avg.** |
| ----------------- | ---------: | ---------: | -------: | -----------: | ------------: | ----------------: | ----------: | -------: |
| *Qwen2.5-7B*      |            |            |          |              |               |                   |             |          |
| GRPO              |       21.2 |        9.6 |     58.7 |         78.8 |          27.9 |              40.7 |        36.7 |     38.6 |
| w. Clip-higher    |       18.1 |       11.5 |     56.6 |         79.2 |          29.8 |              43.3 |        40.4 |     38.8 |
| w. **`CLIP-Cov`** |       22.1 |   **15.8** |     58.2 |         80.4 |      **30.5** |          **44.1** |    **41.1** |     40.4 |
| w. **`KL-Cov`**   |   **22.6** |       12.9 | **61.4** |     **80.8** |          29.1 |              42.6 |        38.2 | **40.6** |
| *Qwen2.5-32B*     |            |            |          |              |               |                   |             |          |
| GRPO              |       21.8 |       16.2 |     69.7 |         84.2 |          35.2 |              43.6 |        45.5 |     45.8 |
| w. Clip-higher    |       35.6 |       22.3 |     69.5 |         77.2 |          35.1 |              42.5 |        43.0 |     47.2 |
| w. **`CLIP-Cov`** |       32.3 |       22.7 |     67.2 |     **87.0** |      **42.0** |          **57.2** |        46.0 |     50.3 |
| w. **`KL-Cov`**   |   **36.8** |   **30.8** | **74.5** |         84.6 |          39.1 |              49.0 |    **46.3** | **52.2** |

Our two approaches both achieve non-trivial improvements across all benchmarks. Compared to GRPO, our method outperforms it by 2.0% on average for the 7B model and by 6.4% for the 32B model. Moreover, we observe that our method yields more substantial gains on the larger Qwen2.5-32B. Specifically, our method achieves improvements of 15.0% and 14.6% compared to GRPO on the most challenging benchmarks, AIME24 and AIME25, respectively.


# 🎈Citation
If you find this paper or repo helpful, please cite us.

```bibtex
@article{cui2025entropy,
  title={The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models},
  author={Cui, Ganqu and Zhang, Yuchen and Chen, Jiacheng and Yuan, Lifan and Wang, Zhi and Zuo, Yuxin and Li, Haozhan and Fan, Yuchen and Chen, Huayu and Chen, Weize and others},
  journal={arXiv preprint arXiv:2505.22617},
  year={2025}
}
```
# Acknowledgement
We implement our reinforcement learning algorithm extending from [verl](https://github.com/volcengine/verl). We utilize [vLLM](https://github.com/vllm-project/vllm) for inference. Our models are trained primarily on [Qwen2.5 family](https://github.com/QwenLM/Qwen2.5). Our training data is built from [hendrycks_math](https://huggingface.co/datasets/hendrydong/hendrycks_math). Thanks for their great contributions!


