# ReneLLM: A Prompt Rewriting Framework (Open Source Version)

ReneLLM is a Python-based framework that implements various prompt rewriting techniques to analyze and transform text prompts using language models. This repository is targetted towards implementing the ReneLLM technique on a carefully chosen set of <a href="https://huggingface.co/">huggingface models </a>.

![](jailbreak.jpg)
## Features

- Multiple prompt rewriting operations:
  - Sentence shortening
  - Sensitive word modification
  - Sentence structure alteration
  - Character insertion
  - Language mixing (Hindi-English)
  - Style transformation

- Automated evaluation pipeline with:
  - Harmful content classification
  - Progress tracking and logging

## Installation

```bash
pip install "numpy<2.0.0"
pip install transformers torch trl
```

# Updations
The distinction of this repository from the original work with ReneLLM is the models that it uses. For the attacker, we utilized `Qwen/Qwen2.5-1.5B-Instruct` model and for the victim, we used `ARahul2003/lamini_flan_t5_detoxify_rlaif` which is a detoxified and a safety aligned model. 

We use `hbseong/HarmAug-Guard` as a judge model to classify whether the generated model contains harmful output or not.

# Evaluation

The evaluation task is to verify the strength of jailbreak prompts to make the victim model vulnerable to generate a corresponding output bypassing its safety alignment capabilities. For thiis purpose, we utilized the `Attack Success Rate (ASR)` metric which quantifies the percentage of successful jailbreak attempts.
