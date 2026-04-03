---
base_model: Qwen/Qwen2-0.5B-Instruct
library_name: transformers
model_name: smoke_test_20260402_181500
tags:
- generated_from_trainer
- trl
- minillm
licence: license
---

# Model Card for smoke_test_20260402_181500

This model is a fine-tuned version of [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with MiniLLM, a method introduced in [MiniLLM: Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543).

### Framework versions

- TRL: 0.29.0
- Transformers: 5.3.0
- Pytorch: 2.10.0
- Datasets: 4.8.2
- Tokenizers: 0.22.2

## Citations

Cite MiniLLM as:

```bibtex
@inproceedings{
    gu2024minillm,
    title={{MiniLLM: Knowledge Distillation of Large Language Models}},
    author={Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=5h0qf7IBZZ}
}
```

Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```