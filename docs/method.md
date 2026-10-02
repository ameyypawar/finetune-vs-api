# Method

How the comparison is measured, and what its numbers can and cannot say. The README tables, the
model card (`hf/README.md`) and the write-up draft (`docs/writeup.md`) are generated from
`results/comparison.json` by `scripts/render.py`; this page is written by hand and defines what is in
them.

## What is compared

Six systems answer the same request, turning it into one JSON function call (an intent and a list of
slots), on the English part of MASSIVE 1.1: the LoRA fine-tune `ft-qwen3-4b-lora`, the untuned base
model with retrieved examples `base-qwen3-4b-k10`, and four API models on free tiers:
`groq-gpt-oss-20b-k10`, `groq-gpt-oss-120b-k10` and `groq-qwen3.8-27b-k10` on Groq, and
`gemini-3.5-flash-lite-k10` on Google AI Studio. `configs/systems.yaml` defines each one: endpoint, model,
prompt, decoding. The fine-tune is the reference: every other system is compared with it.

Thinking settings: the two gpt-oss rows reason at low effort, the lowest they offer, and their
reasoning tokens are reported and priced. `groq-qwen3.8-27b-k10` runs with
thinking off (`reasoning_effort: none`). `gemini-3.5-flash-lite-k10` runs at `minimal`, the least it
accepts (it rejects `none`). Its usage block has no separate count of thinking tokens; at `minimal`, a probe's
total equalled prompt plus completion and took about as long as with more thinking allowed.

## Changes to the plan

- **2026-10-02.** Before any dev or test evaluation request had been sent, the two GitHub Models rows
  (`gh-gpt-4.1-mini-k10` and `gh-gpt-4.1-k10`, openai/gpt-4.1-mini and openai/gpt-4.1) were replaced
  by `groq-gpt-oss-20b-k10` and `groq-qwen3.8-27b-k10` on Groq and `gemini-3.5-flash-lite-k10` on Google AI
  Studio's free tier. GitHub Models stopped taking new customers on 2026-06-16
  (<https://github.blog/changelog/2026-06-16-github-models-is-no-longer-available-to-new-customers/>)
  and was retired on 2026-07-30 (<https://www.developersdigest.tech/blog/github-models-retired-2026>);
  on 2026-10-02 its inference and catalog URLs answered a plain "OK" to every request.
- **2026-10-02, later.** Before any test request, `gemini-3.8-flash-k10` was replaced by
  `gemini-3.5-flash-lite-k10`. gemini-3.8-flash's free tier allows 20 requests per day (its 429 error:
  quotaId `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 20), so S500 would have taken
  25 days. Gemini 3.5 Flash-Lite's free tier is reported at 500 requests per day for September 2026
  (<https://www.scriptbyai.com/gemini-api-free-tier-limits/>). The 14 dev items gemini-3.8-flash answered
  before the cap are not part of the results.
- **2026-10-02, dev selection.** The first dev-select run on Kaggle could not start vLLM: on a T4 it picks
  FlashInfer, whose kernels could not be linked in Kaggle's image (`ld: cannot find -lcuda`), so it fell back to
  plain transformers, one request at a time, which measures accuracy but not throughput. A second run with a
  Triton-attention rung (`vllm-lora-triton`) served every variant with vLLM. Both runs chose epoch 2, with nearly
  the same dev scores (exact match 0.729 and 0.730 for epoch 2; 0.707 and 0.707 for epoch 1; 0.667 and 0.666 for
  the base row): `results/serving/dev_select.json` is the second run, `dev_select_v1_hf_transformers.json` the
  first. Test runs of the self-hosted rows use the second run's serving path.

## Metrics

All of them are computed per item from the parsed answer (`src/finetune_vs_api/metrics.py`).

- **Exact match.** The intent is equal and the multiset of (slot type, value) pairs is equal. Order
  does not matter; duplicates do. A value is compared after lowercasing, trimming and collapsing
  whitespace, and nothing else is normalized: punctuation and word order count.
- **Intent accuracy.** The intent is equal.
- **Slot precision, recall and F1.** Micro-averaged over the items scored. A predicted slot is a true
  positive when its type and its normalized value both match a gold slot, counted as a multiset, so
  predicting the same slot twice when the gold has it once is one true positive and one false positive.
- **Schema-valid rate.** The answer parses as one JSON object (one Markdown code fence around it is
  tolerated, nothing else), has exactly the keys `intent` and `slots`, string types throughout, an intent
  from the 60 in the training split and slot types from its 55. Anything else is a failure to follow the
  format, and it is not repaired.
- **Slot values not in the request.** Among slots from schema-valid answers, the share whose normalized
  value does not occur as text in the normalized request. It measures invented values.

An answer that is not schema-valid, and a call that failed, scores zero on every task metric, and its gold
slots count as false negatives. The schema-valid rate is reported next to the task metrics so a reader can
see how much of a score is lost to format rather than to the task.

### Why this slot F1 is not the MASSIVE paper's

The MASSIVE paper reports a span-based slot score: a slot is credited when its label and its position in
the utterance (its span) match the annotation. The models here answer with text, not with token
positions, so this repository compares (slot type, value text) instead. That differs in ways that move
the number: it cannot tell two occurrences of the same text apart; its normalization (case and
whitespace) is more lenient than an exact span; an invalid answer counts as zero, which a score computed
on tag sequences never has to do; and it is micro-averaged over a fixed subset of the test split, for models
that were not trained or prompted as the paper's were. Do not set these figures beside published slot F1.
Within this repository every system is scored the same way, which is what the comparison needs.

## Subsets and how wide the intervals are

API systems on free tiers have daily request caps, so they cannot see the whole test split. Every system
is scored on a fixed, seeded, scenario-stratified subset (`src/finetune_vs_api/subsets.py`,
`results/subsets.json`):

- **S500**: 500 items of the test split, stratified by MASSIVE's 18 scenarios with proportional
  allocation and a floor of 5 per scenario. This is the headline subset.
- **S300**: 300 items, drawn from inside S500 with the same floor. It is pre-registered, committed in
  `results/subsets.json` with its hash, but no row uses it now: every API row runs on S500.
- **full**: the whole test split (2,974 items). It is a secondary column, for the self-hosted rows only,
  which have no request cap.

Dev has the same design (D50 inside D100) and is used for choosing prompts and the checkpoint. Each
subset has a hash of its item ids, which the test lock records. `scripts/compare.py` stops if `subsets.json` was
edited (its ids no longer match their hash) and warns when a run was made on a different subset than the
current one.

How precise a number is follows from how many items it rests on. A 95% interval on an exact-match
proportion has a half-width of about 1.96 times the square root of p(1-p)/n:

- at 90% exact match: 2.6 points on S500, 3.4 on S300, 1.1 on the full split;
- at 80%: 3.5 on S500, 4.5 on S300, 1.4 on the full split.

A difference between two systems on the same items is tighter than that, and depends on how many items the
two disagree on. With d disagreeing items and the two systems about equally often right on them, the
half-width is about 1.96 times the square root of d, divided by n: 2.5 points for d = 40 on S500, 4.1 for
d = 40 on S300. A gap smaller than its interval is not resolved by these subsets.

## Statistics

- **Intervals.** Percentile bootstraps over items, 10,000 resamples, fixed seed
  (`metrics.BOOTSTRAP_RESAMPLES`, `metrics.BOOTSTRAP_SEED`), so a published interval can be reproduced from the
  predictions.
- **Paired difference.** For each system and the reference, on the items both have an answer for, the
  difference in exact match (system minus reference) with a paired bootstrap: the same resampled items are
  used for both systems, so item difficulty cancels.
- **Exact McNemar test.** Only the items where exactly one system is right carry information. Under the null
  each is equally likely to favour either system, so the smaller count follows a binomial with p = 0.5 and the
  two-sided p-value is twice its lower tail, capped at 1.
- **Wording.** A system "beats" another only when the 95% interval of the paired difference excludes 0.
  Otherwise the comparison is reported as "no significant difference". The McNemar p-value is shown beside
  it; when the two disagree about significance, `scripts/compare.py` records a warning and the wording
  follows the interval.

The comparison is one run per system at temperature 0. It does not measure run-to-run variance, and a
model name that changes during a run (an endpoint silently switching versions) is flagged as a warning, since
the run is then a mixture.

## Cost

**API rows** run on free tiers, so no money is spent. Every cost is what the same calls would cost at the
provider's paid list price (`free tier; priced at paid list price`), from the entries in
`configs/sources.yaml`, each with the page it was read from and the date.

For each call the tokens come from the provider's usage report (`prompt_tokens` includes any cached
tokens; `completion_tokens` includes reasoning tokens). Because a provider's prompt caching may or may not
apply, the cost is a pair of bounds, per 1,000 calls (`src/finetune_vs_api/cost.py`):

- **no caching**: every prompt token at the full input price;
- **cached prefix**: the static part of the prompt (the system message, identical for every request) at the
  cached input price.

A price entry with no cached input price (qwen3.8-27b on Groq lists none) gets no caching discount, so
its two bounds are equal.

**Self-hosted rows** are priced as a rented GPU: the on-demand hourly price of the `aws-g4dn.xlarge` entry (one
T4), divided by the requests per second the throughput benchmark measured at the operating point. It is the
cost with the GPU kept busy; an idle or half-used GPU costs proportionally more.

**Break-even** is the monthly API volume at which the API bill equals the cost of keeping that GPU for
every hour of the month: the hourly price times 730 hours, divided by the API's cost per call, at each bound.
Above that volume the rental is cheaper, provided one GPU can serve it. `scripts/compare.py` therefore also
records what one GPU serves a month at the operating point (requests per second times 3,600 times 730) and
whether each break-even volume fits within it; where it does not, the GPU never gets cheaper on this
request mix.

What this leaves out: the throughput is measured on a Kaggle T4 and assumed to carry over to the rented
T4 (the same GPU class, not measured on AWS); spot prices; engineering time, storage and network; bursty or
low traffic; more than one GPU. The API prompts are long (retrieved examples) and the fine-tune's is a
single line, which is part of what is being compared, not an artefact.

## Latency

- **Headline: the self-hosted rows, on the box.** `scripts/bench_throughput.py` runs next to the server, so
  no network is in the path, and writes `results/serving/<gpu>.json`. The headline numbers are p50 and p95 at
  concurrency 1 and p95 at the operating point, which is the highest concurrency whose p95 stays under the
  benchmark's limit with no failed request. Percentiles use the nearest-rank rule.
- **Appendix: the API rows, as observed.** The runner records each call's latency for the successful attempt,
  excluding waits for a rate limit and retry backoff. These are *observed on free tiers from India; not
  representative of paid tiers*, and they are only ever shown with that label. They are not compared with the
  self-hosted figures.

`scripts/compare.py` reads, for the GPU named in the rental entry, a file with `gpu`, `system`, `levels` (each
with `concurrency`, `requests_per_s` and `latency_s.p50` and `.p95`, in seconds) and `operating_point`. A file
that does not fit is reported in `warnings`, never guessed at.

## The test lock

The test split must not be tuned against, so nothing may touch it before its configuration is frozen.

1. Choose prompts and the checkpoint on the dev split (`dev_prompts` in `configs/systems.yaml`; D50 and D100).
2. Pin the checkpoint (adapter, epoch, base revision) in `configs/systems.yaml`. `python scripts/lock_test.py
   --show` lists what still blocks locking a system.
3. `python scripts/lock_test.py --write --system NAME --reason "why now"` appends an entry to
   `results/test_lock.jsonl`: hashes of the rendered prompt, the output schema, the endpoint and model, the
   decoding, the checkpoint, the price entry and the dataset, plus the hash of the test subset, the reason,
   the time and the git commit. Rate limits and concurrency are not hashed: they change how fast a run goes,
   not what it measures.
4. `python scripts/run_eval.py --system NAME --split test` checks the lock before reading any test data or
   sending any request, and refuses unless the current configuration and subset match a lock.
5. Any later change to something hashed needs a new lock with a reason. The history is append-only and tracked
   in git, and each run summary records the lock it ran under. `scripts/compare.py` warns about a run whose
   summary records none.

The lock makes tuning on the test split visible; it cannot stop someone from looking at test results and
locking again.

## Contamination and the data

- MASSIVE has been public since 2022, and an MTEB mirror redistributes the test text. The base model and the
  API models may have seen it in pretraining. That cannot be measured here and would favour every system, in
  unknown amounts.
- The data repeats. At the pinned archive, 21 of the 2,974 test items have text that also occurs in the train
  split (31 if punctuation is ignored), and 46 request texts occur twice within train, five of them with
  different labels. A repeated request is a gift to two of the systems: the fine-tune trained on it, and
  few-shot retrieval can return the identical request with its label. `results/data_audit.json` lists them.
  Exact match on the test items whose text never occurs in train is reported next to the headline figure, using
  the same lowercased, trimmed, whitespace-collapsed text the metrics use.
- The label inventory comes from the train split only, and the fine-tune is trained on MASSIVE's human labels
  only; no API output is ever a label.
- Free-tier data use. Google may use content sent on its free tier to improve its products. Only public MASSIVE
  text is sent to any API: the requests, and in the few-shot prompts the label lists and train examples taken
  from MASSIVE.

## What the scripts read and write

- `scripts/compare.py` reads the test predictions and summaries under `results/runs/`, the gold labels in
  `data/processed/`, `results/data_audit.json`, the benchmark file and `configs/`, and writes
  `results/comparison.json`. Missing inputs are reported under `warnings`, never guessed; with no results it
  writes a file saying so.
- `scripts/make_figures.py` writes `results/figures/accuracy_vs_cost.png` (exact match against cost per 1,000
  calls on a logarithmic axis) and `results/figures/latency.png` (self-hosted latency only). It draws to files
  with matplotlib's Agg backend, so it needs no display.
- `scripts/render.py --target readme|card|writeup|all` fills the region of `README.md` between its
  `results:start` and `results:end` markers, writes `hf/README.md` and `docs/writeup.md`, and with `--check`
  exits 1 if any of them is out of date. CI runs the check. With no results the README is left byte-for-byte
  as committed.
  Every number comes from a file and none is typed into a template. Every table is followed by the same
  notice, with the source URLs and the dates they were read on: all API rows ran on free tiers, no money was
  spent, and costs are at paid list prices.
- Optional inputs that the card and write-up use when they exist: `results/train_log.json` (written by
  `kaggle/train_on_kaggle.py`), `results/error_analysis.csv` (one row per hand-labelled error; a `category`
  column is counted) and `docs/error_analysis.md` (the hand-written reading, included verbatim in the
  write-up so re-rendering never overwrites it).

Figures are drawn from, and the generated files are checked against, what is committed: commit
`results/comparison.json` and `results/figures/` together with the files rendered from them.
