# Notice

This project uses a third-party dataset and a third-party model. This file records what is
used, under which licence, and what was changed. The project's own code is MIT licensed
(see `LICENSE`). It is an independent project and is not endorsed by Amazon, the SLURP
authors or the Qwen team.

## Data: MASSIVE 1.1 (English, en-US)

The task data is derived from the English (en-US) part of MASSIVE 1.1, released by Amazon
under the Creative Commons Attribution 4.0 International licence (CC BY 4.0,
<https://creativecommons.org/licenses/by/4.0/>). The dataset's own licence file begins
"Copyright Amazon.com Inc. or its affiliates."

Please cite MASSIVE as:

```
@misc{fitzgerald2022massive,
      title={MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
      author={Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      year={2022},
      eprint={2204.08582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

MASSIVE: <https://arxiv.org/abs/2204.08582>

## Data: SLURP

MASSIVE was created by translating and localizing the English text of SLURP, and some of
SLURP's text is included in MASSIVE as it is. SLURP is also licensed CC BY 4.0. Please cite
SLURP as:

```
@inproceedings{slurp,
    author = {Emanuele Bastianelli and Andrea Vanzo and Pawel Swietojanski and Verena Rieser},
    title={{SLURP: A Spoken Language Understanding Resource Package}},
    booktitle={{Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)}},
    year={2020}
}
```

SLURP: <https://aclanthology.org/2020.emnlp-main.588/>

## Changes made to the data

Annotations converted to JSON. MASSIVE marks slots inline in the request, as
`[label : value]`. `scripts/prepare_data.py` reads those spans and writes each example as a
JSON function call, `{"intent": ..., "slots": [{"type": ..., "value": ...}]}`, with the slots
in order of appearance, and as OpenAI chat-format records (`sft_train.jsonl`,
`sft_dev.jsonl`). MASSIVE's own train / dev / test partition is kept as it is. The
converted files carry the same CC BY 4.0 licence and this attribution. The data itself is
not committed to this repository: it is downloaded from the canonical archive and verified
against the sha256 pinned in `configs/data.yaml`.

## Model: Qwen3-4B-Instruct-2507, and the adapter to be released later

The base model, `Qwen/Qwen3-4B-Instruct-2507`, is published by the Qwen team under the
Apache License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>). This repository
does not contain or redistribute its weights.

A LoRA adapter trained on top of that model is planned for release later. It is a modification
of the base model's behaviour, and only works together with the base model. When it is
released it will:

- ship with a copy of the Apache License 2.0 and this notice;
- say that it was changed: fine-tuned with LoRA on the MASSIVE-derived data above;
- name the base model and the exact commit it was trained from (`configs/train.yaml`);
- carry the MASSIVE and SLURP attribution above, because it was trained on that data.

The licence chosen for the adapter's own files will be stated with the release.
