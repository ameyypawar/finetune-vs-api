---
license: apache-2.0
base_model: "Qwen/Qwen3-4B-Instruct-2507"
library_name: peft
pipeline_tag: text-generation
language:
- en
datasets:
- "AmazonScience/massive"
tags:
- lora
- peft
- function-calling
- intent-detection
- slot-filling
model-index:
- name: "ft-qwen3-4b-lora"
  results:
  - task:
      type: text-generation
      name: Text generation
    dataset:
      name: "MASSIVE 1.1 en-US, test split (n=2974)"
      type: "AmazonScience/massive"
      config: "en-US"
      split: test
    metrics:
    - type: "exact_match"
      name: "Exact match"
      value: 0.7327
    - type: "accuracy"
      name: "Intent accuracy"
      value: 0.9001
    - type: "f1"
      name: "Slot F1"
      value: 0.8257
  - task:
      type: text-generation
      name: Text generation
    dataset:
      name: "MASSIVE 1.1 en-US, test subset S500 (n=500)"
      type: "AmazonScience/massive"
      config: "en-US"
      split: test
    metrics:
    - type: "exact_match"
      name: "Exact match"
      value: 0.75
    - type: "accuracy"
      name: "Intent accuracy"
      value: 0.91
    - type: "f1"
      name: "Slot F1"
      value: 0.8401
---

# ft-qwen3-4b-lora

A LoRA adapter for [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that turns a free-text request into one JSON function call: an intent, and a list of slots whose values are copied from the request. It was trained on the English (`en-US`) part of MASSIVE 1.1, which has 60 intents and 55 slot types.

```
"wake me up at nine am on friday"
  -> {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "friday"}]}
```

The adapter only works together with its base model. It is not a general chat model. It was compared with API models in the finetune-vs-api repository, whose `docs/method.md` defines every metric below.

## Use

The prompt must be the one it was trained with: a system message holding the one-line instruction below, then the request as the user message, with nothing added.

### With vLLM

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --lora-modules ft-qwen3-4b-lora=<adapter repo id or local path> \
  --max-lora-rank 16
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="ft-qwen3-4b-lora",
    messages=[
        {"role": "system", "content": "Convert the request into a JSON function call with an intent and slots."},
        {"role": "user", "content": "wake me up at nine am on friday"},
    ],
    temperature=0,
    max_tokens=256,
)
print(reply.choices[0].message.content)
```

### With transformers and peft

```python
import json

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "<adapter repo id or local path>")
model.eval()

messages = [
    {"role": "system", "content": "Convert the request into a JSON function call with an intent and slots."},
    {"role": "user", "content": "wake me up at nine am on friday"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(json.loads(reply))
```

Older transformers releases call the `dtype` argument `torch_dtype`. Decoding is greedy (temperature 0). The reply is parsed as plain JSON: text around the object, or more than one object, counts as a failure to follow the format.

The adapter location above is a placeholder until the adapter is published.

## Training details

Read from the training log (`results/train_log.json`, written by `kaggle/train_on_kaggle.py`).

- Base model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Method: LoRA with rank 16, alpha 32 and dropout 0.0, on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj; the base weights were quantized while training (`load_in_4bit`, QLoRA).
- Data: the train split of MASSIVE (11,514 requests) with its human labels, converted to chat records. Labels never come from an API model.
- Optimisation: learning rate 0.0002, linear schedule with warmup ratio 0.03, weight decay 0.01, optimizer adamw_8bit, 2 epochs, batch size 8 with 2 gradient accumulation steps, maximum sequence length 512, seed 3407.
- Precision: fp16.
- Hardware: Tesla T4, 48.2 minutes.
- Final training loss 0.0954; validation loss 0.0605 after epoch 1, 0.0527 after epoch 2.
- An adapter was saved after each of the epochs 1, 2; epoch 2 was chosen on the dev split.
- Training files: `sft_train.jsonl` (11,514 records, sha256 `97418d8f03494a2b4422dc4745237cec5d886890ac8f995e32547f76eb24eb9c`); `sft_dev.jsonl` (2,033 records, sha256 `1dd2d0e5d6cfa48f21c868448d36c194a703915d17252a60db485f2e788b8996`).
- Packages: unsloth 2026.9.12, unsloth_zoo 2026.9.8, trl 0.24.0, transformers 5.5.0, peft 0.21.1, accelerate 1.15.0, bitsandbytes 0.50.2, datasets 4.3.0, torch 2.10.0+cu128, xformers 0.0.35, tokenizers 0.22.2, huggingface-hub 1.11.0.

## Evaluation

Each system is scored on a fixed, seeded test subset stratified by scenario: S500 (500 items), or a smaller nested one where the table says so. The self-hosted rows are also scored on the full test split. Intervals are 95% percentile bootstraps. The difference is the system minus the fine-tune, with a paired bootstrap interval and an exact McNemar test; a system "beats" another only when that interval excludes zero.

| System | Items | Exact match | Difference from the fine-tune | McNemar p | Reading | Full test split |
|---|---|---:|---:|---:|---|---:|
| `ft-qwen3-4b-lora` | S500 (500) | 75.0% [71.2, 78.8] | reference | - | reference | 73.3% [71.7, 74.8] (n=2974) |
| `base-qwen3-4b-k10` | S500 (500) | 66.6% [62.4, 70.8] | -8.4 pp [-12.4, -4.6] | <0.001 | the fine-tune beats it | 66.1% [64.4, 67.8] (n=2974) |
| `groq-gpt-oss-20b-k10` | S500 (500) | 62.2% [57.8, 66.4] | -12.8 pp [-16.6, -9.0] | <0.001 | the fine-tune beats it | - |
| `groq-gpt-oss-120b-k10` | S500 (500) | 63.0% [58.8, 67.4] | -12.0 pp [-16.0, -8.2] | <0.001 | the fine-tune beats it | - |
| `groq-qwen3.8-27b-k10` | S500 (500) | 70.8% [66.8, 74.8] | -4.2 pp [-7.6, -0.6] | 0.024 | the fine-tune beats it | - |
| `gemini-3.5-flash-lite-k10` | S500 (500) | 67.8% [63.6, 71.8] | -7.2 pp [-10.8, -3.6] | <0.001 | the fine-tune beats it | - |

*All API rows ran on free tiers; no money was spent; costs are at paid list prices.*

The fine-tune's exact match has a 95% interval of plus or minus 3.8 pp on S500 and 1.6 pp on the full split.

### Serving cost and latency

Measured on the box, with no network in the path, at every load level fixed in advance. No level met the rule for an operating point (p95 latency at or under 1 s), so cost is reported at every measured level instead. Each cost is a GPU rented at that price and kept busy at that load, at the on-demand price and at the spot price; the calls one GPU serves a month are its throughput at that load over 730 hours.

| Concurrency | Requests/s | p50 | p95 | Cost per 1,000 calls, on-demand | Cost per 1,000 calls, spot | Calls one GPU serves a month | APIs whose break-even range one GPU can serve |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.79 | 1.26 s | 2.27 s | $0.185 | $0.0963 | 2,076,901 | `groq-gpt-oss-120b-k10`, `groq-qwen3.8-27b-k10`, `gemini-3.5-flash-lite-k10` |
| 8 | 5.39 | 1.47 s | 2.57 s | $0.0271 | $0.0141 | 14,152,669 | `groq-gpt-oss-20b-k10`, `groq-gpt-oss-120b-k10`, `groq-qwen3.8-27b-k10`, `gemini-3.5-flash-lite-k10` |
| 32 | 14.40 | 2.16 s | 3.69 s | $0.0101 | $0.0053 | 37,837,287 | `groq-gpt-oss-20b-k10`, `groq-gpt-oss-120b-k10`, `groq-qwen3.8-27b-k10`, `gemini-3.5-flash-lite-k10` |
| 64 | 21.61 | 2.86 s | 4.67 s | $0.0068 | $0.0035 | 56,800,731 | `groq-gpt-oss-20b-k10`, `groq-gpt-oss-120b-k10`, `groq-qwen3.8-27b-k10`, `gemini-3.5-flash-lite-k10` |

*All API rows ran on free tiers; no money was spent; costs are at paid list prices.*

*Prices: <https://console.groq.com/docs/model/openai/gpt-oss-20b> (retrieved 2026-10-02), <https://console.groq.com/docs/model/openai/gpt-oss-120b> (retrieved 2026-10-01), <https://console.groq.com/docs/model/qwen/qwen3.8-27b> (retrieved 2026-10-02), <https://ai.google.dev/gemini-api/docs/pricing> (retrieved 2026-10-02). GPU rental: <https://instances.vantage.sh/aws/ec2/g4dn.xlarge> (retrieved 2026-10-01).*

## Limitations

- English only: the training and the evaluation use the `en-US` part of MASSIVE. Other languages, other domains and spoken input were not tested.
- The labels are fixed: 60 intents and 55 slot types. A request outside them still gets one of them.
- Slot values are meant to be copied from the request, but a model can emit a value that is not in it; the share of predicted slot values missing from the request is in `results/comparison.json`. Nothing constrains the decoding by default, so the output can fail to follow the JSON format; the schema-valid rate is reported next to the task metrics.
- MASSIVE has been public since 2022, and an MTEB mirror of it redistributes the test text, so the base model may have seen the test items. The data also repeats: 21 test items have text that also occurs in train, which the evaluation reports separately.
- The comparison covers one task, one run per system and subsets of the test split, so small differences are not resolved. `docs/method.md` has the details, including how throughput and cost were measured.

## License and attribution

The adapter is released under the license named in the header. The base model, Qwen/Qwen3-4B-Instruct-2507, is published by the Qwen team under Apache-2.0 and is not part of this repository. This is an independent project, not endorsed by Amazon, the SLURP authors or the Qwen team.

Change made: the base model was fine-tuned with LoRA on data derived from MASSIVE 1.1, in which MASSIVE's `[label : value]` annotations were converted to JSON function calls, keeping MASSIVE's train, dev and test partition.

The data is released by Amazon under CC-BY-4.0. Its English text comes from SLURP, which carries the same license (see `NOTICE.md` in the repository). Please cite both.

```bibtex
@misc{fitzgerald2022massive,
      title={MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
      author={Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      year={2022},
      eprint={2204.08582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@inproceedings{slurp,
    author = {Emanuele Bastianelli and Andrea Vanzo and Pawel Swietojanski and Verena Rieser},
    title={{SLURP: A Spoken Language Understanding Resource Package}},
    booktitle={{Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)}},
    year={2020}
}
```
