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
---

# ft-qwen3-4b-lora

A LoRA adapter for [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that turns a free-text request into one JSON function call: an intent, and a list of slots whose values are copied from the request. It was trained on the English (`en-US`) part of MASSIVE 1.1, which has 60 intents and 55 slot types.

```
"wake me up at nine am on friday"
  -> {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "friday"}]}
```

The adapter only works together with its base model. It is not a general chat model. It was compared with API models in the finetune-vs-api repository, whose `docs/method.md` defines every metric below.

**Evaluation results are not in yet.** This card is generated from the repository's results directory, so the table and the metadata above fill in once the test runs have been compared.

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

No training run has finished yet. These are the planned settings from `configs/train.yaml`, not the record of a run.

- Base model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Method: LoRA with rank 16, alpha 32 and dropout 0.0, on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj; the base weights were quantized while training (`load_in_4bit`, QLoRA).
- Data: the train split of MASSIVE (11,514 requests) with its human labels, converted to chat records. Labels never come from an API model.
- Optimisation: learning rate 0.0002, linear schedule with warmup ratio 0.03, weight decay 0.01, optimizer adamw_8bit, 2 epochs, batch size 8 with 2 gradient accumulation steps, maximum sequence length 512, seed 3407.

## Evaluation

The evaluation compares this adapter with the untuned base model and with API models on a fixed test subset, using paired bootstrap intervals and an exact McNemar test. The table will appear here once the test runs exist.

## Limitations

- English only: the training and the evaluation use the `en-US` part of MASSIVE. Other languages, other domains and spoken input were not tested.
- The labels are fixed: 60 intents and 55 slot types. A request outside them still gets one of them.
- Slot values are meant to be copied from the request, but a model can emit a value that is not in it. Nothing constrains the decoding by default, so the output can fail to follow the JSON format; the schema-valid rate is reported next to the task metrics.
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
