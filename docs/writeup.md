# A small fine-tune against API models, on one narrow task

## The question

Can a LoRA fine-tune of a small open model match API models on a narrow, well-specified task, and what does each cost to run? The task is to turn a request into one JSON function call: an intent (one of 60) and slots (each one of 55 types, its value copied from the request). The data is the English part of MASSIVE 1.1: 11,514 train, 2,033 dev and 2,974 test requests.

## The setup

The comparison has 6 systems:

- `ft-qwen3-4b-lora`: Qwen/Qwen3-4B-Instruct-2507 plus the LoRA adapter, prompt `finetuned_v1` (one-line instruction), self-hosted.
- `base-qwen3-4b-k10`: Qwen/Qwen3-4B-Instruct-2507, prompt `fewshot_k10_v1` (10 retrieved examples), self-hosted.
- `groq-gpt-oss-20b-k10`: openai/gpt-oss-20b, prompt `fewshot_k10_v1` (10 retrieved examples), Groq, free tier.
- `groq-gpt-oss-120b-k10`: openai/gpt-oss-120b, prompt `fewshot_k10_v1` (10 retrieved examples), Groq, free tier.
- `groq-qwen3.8-27b-k10`: qwen/qwen3.8-27b, prompt `fewshot_k10_v1` (10 retrieved examples), Groq, free tier.
- `gemini-3.8-flash-k10`: gemini-3.8-flash, prompt `fewshot_k10_v1` (10 retrieved examples), Google AI Studio, free tier.

The fine-tune is a LoRA adapter (rank 16, alpha 32, 2 epochs, as planned: no training run has finished yet) fitted to the human-labelled train split only; no API output is a label. The API systems run on free tiers with daily request caps, so every system is scored on a fixed, seeded test subset, stratified by scenario: S500 (500 items), S300 (300 items, inside S500). Each system's configuration must be locked, with a reason, before the runner will touch the test split. No money is spent on the API systems: their costs are computed at paid list prices.

## Results

No test run has been compared yet. Once the runs exist, `scripts/compare.py` writes the numbers and this section shows a table with each system's exact match and a 95% interval, the difference from the fine-tune with a paired interval, and a figure of exact match against cost per 1,000 calls on a logarithmic axis. Every system is scored on the same fixed subsets, so a difference is a difference on the same items, not on different samples. The wording rule is fixed in advance: a system beats another only where the interval of the paired difference excludes zero, and anything else is reported as no significant difference.

## Where the gap comes from

Exact match needs the intent and every slot right. Once results exist, this section splits it into intent accuracy, slot F1, the schema-valid rate and the share of slot values that are not in the request, gives exact match without the test items whose text also occurs in train, and names the fine-tune's weakest scenario. The error analysis that follows is written by hand from sampled errors.

### Error analysis

This is the slot for the hand-labelled error analysis: the errors go in `results/error_analysis.csv`, with a `category` column that is counted here, and the reading of them goes in `docs/error_analysis.md`, which is included here. Until it is written, the draft makes no claim about why the systems differ.

## Cost and break-even

API costs are computed from tokens at the providers' paid list prices, as a no-caching bound and a bound with the static prompt prefix cached. Self-hosted cost comes from throughput measured on Kaggle's free T4, priced at the on-demand rate for renting the same GPU on AWS (g4dn.xlarge). The break-even is the monthly volume at which a GPU rented around the clock (730 hours at the on-demand price) costs less than each API, given that one GPU can serve that volume. The self-hosted figure is the price with the GPU kept fully busy; an idle or half-used GPU costs proportionally more. Self-hosted latency is measured on the box: p50 and p95 for a single stream of requests, and p95 at the operating point. API latency goes to the appendix.

## Why now

OpenAI's deprecations page (https://developers.openai.com/api/docs/deprecations, retrieved 2026-10-01) lists these dates for fine-tuned models, evals and new fine-tuning jobs:

- On 2026-10-23: Fine-tuned models shut down: ft-gpt-3.5-turbo, ft-gpt-4, ft-gpt-4.1-nano-2025-04-14, ft-babbage-002, ft-davinci-002 and ft-o4-mini-2025-04-16.
- From 2026-10-31: Evals become read-only.
- On 2026-11-30: Evals shut down.
- From 2027-01-06: Active existing customers can no longer create new fine-tuning jobs. Inference on existing fine-tuned models continues until their base model is deprecated.

An open-weights fine-tune that its owner serves does not depend on a provider's schedule.

## Limits

- The test subsets are samples: small differences are not resolved.
- Each system runs once, at temperature 0, so run-to-run variance is not measured.
- API rows run on free tiers and are priced at list prices, which can change; their latency, where shown, is observed on free tiers from India; not representative of paid tiers.
- Self-hosted cost assumes the throughput measured on the benchmark GPU carries over to the rented GPU.
- MASSIVE has been public since 2022 and an MTEB mirror redistributes its test text, so any model may have seen it; 21 test items also occur in train.

## Reproduce

```bash
python scripts/prepare_data.py
python scripts/make_subsets.py
python scripts/lock_test.py --write --system NAME --reason "why now"
python scripts/run_eval.py --system NAME --split test
python scripts/bench_throughput.py --help
python scripts/compare.py && python scripts/make_figures.py
python scripts/render.py --target all
```

Definitions are in `docs/method.md`.

## Appendix: API latency

API latency will be listed here once the runs exist: observed on free tiers from India; not representative of paid tiers.
