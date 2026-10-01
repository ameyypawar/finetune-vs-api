# finetune-vs-api

A LoRA fine-tune of a small open model, compared with API models, on one narrow task:
turning a free-text request into a single JSON function call.

```
"wake me up at nine am on friday"
  -> {"intent": "alarm_set", "slots": [{"type": "time", "value": "nine am"}, {"type": "date", "value": "friday"}]}
```

The intent is one of 60 and each slot type is one of 55; each slot value is text copied from
the request. The data is the English (en-US) part of MASSIVE 1.1 (11,514 train, 2,033 dev,
2,974 test requests).

**Status: work in progress, with no results yet.** The data pipeline, metrics, cost arithmetic,
test lock, API client and evaluation runner exist and are tested against stubs. The Kaggle
training script is written but has not been run. No model has been trained, no API has been
called, and nothing below the "Results" heading has been produced.

## What will be compared

| System | Model | Prompt | Where it runs |
|---|---|---|---|
| `ft-qwen3-4b-lora` | Qwen3-4B-Instruct-2507 + LoRA adapter | `finetuned_v1` | local server |
| `base-qwen3-4b-k10` | Qwen3-4B-Instruct-2507 | `fewshot_k10_v1` | local server |
| `gh-gpt-4.1-mini-k10` | openai/gpt-4.1-mini | `fewshot_k10_v1` | GitHub Models, free tier |
| `gh-gpt-4.1-k10` | openai/gpt-4.1 | `fewshot_k10_v1` | GitHub Models, free tier |
| `groq-gpt-oss-120b-k10` | openai/gpt-oss-120b | `fewshot_k10_v1` | Groq, free tier |

`fewshot_k10_v1` shows the model the 10 most similar training examples (embedding search with
`BAAI/bge-small-en-v1.5`, over the train split only). The fine-tune is trained on MASSIVE's
human-labelled train split; no API output is ever used as a label. Everything is defined in
`configs/` and `src/finetune_vs_api/prompts.py`.

## Method

- **Metrics.** Exact match (intent and the multiset of slots), intent accuracy, slot
  precision/recall/F1, the share of outputs that are valid under the schema, and the share of
  predicted slot values that do not appear in the request. Slot values are compared lowercased
  and whitespace-collapsed; nothing else is normalized. An output that is not schema-valid
  scores zero on every task metric, and the schema-valid rate is reported next to them.
  Intervals are 95% percentile bootstraps (10,000 resamples, fixed seed).
- **Fixed subsets.** API systems run on free tiers with daily request caps, so they cannot see
  the whole test split. Every system runs on a fixed, seeded, scenario-stratified subset:
  D50 inside D100 (dev), S300 inside S500 (test). They are listed, with hashes, in
  `results/subsets.json`.
- **Test lock.** `scripts/lock_test.py` freezes a system's configuration (prompt, schema,
  model, decoding, checkpoint, price entry, dataset) and its test subset, with a reason, in an
  append-only history. `scripts/run_eval.py --split test` refuses to start unless the lock
  matches, so the test split cannot be tuned against.
- **Cost.** The API systems are not billed. Costs are what the same calls would cost at the
  providers' paid list price (`free tier; priced at paid list price`), given as a pair of
  bounds: with no prompt caching, and with the static prompt prefix cached. Prices, limits and
  the dates and pages they were read from are in `configs/sources.yaml`.
- **Free-tier caps.** When a daily quota runs out the runner saves its progress and stops,
  saying when the quota resets; run the same command again with `--resume` and it carries on.
  Nothing already answered is sent twice.

## Reproduce what exists so far

Needs `uv` and Python 3.11. Nothing here uses a GPU, an API key or a model weight (the
embedding model, about 65 MB, is only downloaded the first time a few-shot prompt is built).

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e . --no-deps

pytest -q
ruff check .

python scripts/prepare_data.py       # downloads the archive (about 40 MB), verifies its sha256,
                                     # writes data/processed/ and results/data_audit.json
python scripts/make_subsets.py       # D50 / D100 / S300 / S500 -> results/subsets.json
python scripts/lock_test.py --show   # what each system still needs before it can be locked
python scripts/validate_chat_jsonl.py data/processed/sft_train.jsonl
python kaggle/train_on_kaggle.py --check data/processed   # validates the training files; no GPU
```

To bring your own OpenAI fine-tuning file, run `scripts/validate_chat_jsonl.py` on it first: it
accepts single-turn `{"messages": [system?, user, assistant]}` records and reports tools,
multimodal parts, weights and multi-turn records as unsupported in this version.

Running an evaluation needs an endpoint, which nothing in this repository has contacted yet:

```bash
cp .env.example .env                 # then fill in the keys you have
python scripts/check_free_tiers.py --dry-run
python scripts/run_eval.py --system gh-gpt-4.1-mini-k10 --split dev --subset D50
```

## Know your data

`results/data_audit.json` has the details. In short: MASSIVE repeats some requests, within train
and across splits (a few repeated train requests even carry different labels), and a handful of
intents and slot types never occur in dev or test. These are facts about the data, not results.
The audit reports them so a reader can judge how much they matter.

## Layout

```
configs/      data, training, prices and limits, systems
src/finetune_vs_api/   data, schema, metrics, cost, config and test lock, prompts,
                       retrieval, subsets, client, evaluate
scripts/      prepare_data, make_subsets, lock_test, run_eval, check_free_tiers, validate_chat_jsonl
kaggle/       the training script and metadata (written, not run)
results/      the audit, subsets, lock history, free-tier checks, runs
tests/
```

## Results

Nothing yet. The tables will be generated from `results/` and placed between the markers.

<!-- results:start -->
<!-- results:end -->

## Licence and attribution

Code: MIT (`LICENSE`). The data is derived from MASSIVE and SLURP (CC BY 4.0), and the base
model is Qwen3-4B-Instruct-2507 (Apache 2.0): see `NOTICE.md` for the citations, what was
changed, and what will accompany the adapter when it is released.
