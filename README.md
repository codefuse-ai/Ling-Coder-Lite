# Ling-Coder-Lite

<p align="center">
    <img src="./figures/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://github.com/codefuse-ai/Ling-Coder-Lite">GitHub</a>

## Introduction

Ling-Coder-Lite is a MoE LLM provided and open-sourced by InclusionAI, which has 16.8 billion parameters with 2.75 billion activated parameters. Ling-Coder-Lite performs impressively on coding tasks compared to existing models in the industry. Specifically, Ling-Coder-Lite further pre-training from an intermediate checkpoint of Ling-Lite, incorporating an additional 3 trillion tokens. This extended pre-training significantly boosts the coding abilities of Ling-Lite, while preserving its strong performance in general language tasks.

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.

<div align="center">

|   **Model**    | **#Total Params** | **#Activated Params** | **Context Length** |                                                                     **Download**                                                                     |
| :------------: | :---------------: | :-------------------: | :----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: |
| Ling-lite-base |       16.8B       |         2.75B         |        64K         | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-base) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-lite-base) |
|   Ling-lite    |       16.8B       |         2.75B         |        64K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-lite)      |
| Ling-plus-base |       290B        |         28.8B         |        64K         | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-plus-base) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-plus-base) |
|   Ling-plus    |       290B        |         28.8B         |        64K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-plus) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-plus)      |
| Ling-coder-lite-base |       16.8B        |         2.75B         |        16K         | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-Coder-lite-base) <br>[🤖 ModelScope](https://modelscope.cn/models/inclusionAI/Ling-Coder-lite-base) |
|   Ling-code-lite    |       16.8B        |         2.75B         |        16K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-Coder-lite) <br>[🤖 ModelScope](https://modelscope.cn/models/inclusionAI/Ling-Coder-lite)      |

</div>

## Evaluation

Detailed evaluation results are reported in our [technical report](https://github.com/codefuse-ai/Ling-Coder-Lite/blob/master/Ling_Coder_Lite_Technique_Report.pdf).

## Quickstart

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-Coder-lite"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

prompt = "Write a quick sort algorithm in python."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### 🤖 ModelScope

If you're in mainland China, we strongly recommend you to use our model from 🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>.

## Deployment

### vLLM

vLLM supports offline batched inference or launching an OpenAI-Compatible API Service for online inference.

#### Environment Preparation

Since the Pull Request (PR) has not been submitted to the vLLM community at this stage, please prepare the environment by following the steps below:

```bash
git clone -b  v0.7.3 https://github.com/vllm-project/vllm.git
cd vllm
git apply Ling-Coder-Lite/inference/vllm/bailing_moe.patch
pip install -e .
```

#### Offline Inference:

```bash
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-Coder-lite")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

llm = LLM(model="inclusionAI/Ling-Coder-lite", dtype='bfloat16')
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling-Coder-Lite, an assistant created by CodeFuse-AI"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)


```

#### Online Inference:

```bash
vllm serve inclusionAI/Ling-lite \
              --tensor-parallel-size 2 \
              --pipeline-parrallel-size 1 \
              --use-v2-block-manager \
              --gpu-memory-utilization 0.90
```

For detailed guidance, please refer to the vLLM [`instructions`](https://docs.vllm.ai/en/latest/).

## Finetuning

We recommend you to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to finetune Ling with SFT, DPO, etc.

We use [`identity`](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json) to demonstrate how to finetune our Ling models by replacing `name` with `Ling` and `author` with `inclusionAI`.

```json
{
  "instruction": "hi",
  "input": "",
  "output": "Hello! I am Ling-Coder-Lite, an AI assistant developed by CodeFuse-AI. How can I assist you today?"
}
```

We provide a demo configuration of `Llama-Factory` to SFT Ling models as follows:

```bash
llamafactory-cli train examples/sft/ling_full_sft.yaml
```

## License

This code repository is licensed under [the MIT License](https://github.com/codefuse-ai/Ling-Coder-Lite/blob/master/LICENCE).

## Citation

[TBD]
