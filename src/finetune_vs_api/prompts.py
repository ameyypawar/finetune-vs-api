"""Versioned prompt registry.

A prompt is a pure function of (request, label inventory, retrieved examples) to a list of
chat messages. Prompts are never edited in place: a change gets a new name (`..._v2`), and
`prompt_hash` fingerprints what a name renders, so a silent edit shows up in the test lock.

Layout, for every prompt: everything that is the same for all requests comes first (the
system message), and everything that varies comes last (retrieved examples, then the
request). That ordering is what lets a provider's automatic prompt caching reuse the prefix.

The user turn is always the bare request, in every prompt, so the systems differ in the
instructions and examples they are given and not in how the request is wrapped.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .data import Example, Slot
from .schema import LabelInventory, target_json

FINETUNED_INSTRUCTION = "Convert the request into a JSON function call with an intent and slots."

_ZEROSHOT_TEMPLATE = """\
You convert a request to a voice assistant into exactly one JSON function call.

Reply with a single JSON object and nothing else:
{{"intent": "<intent>", "slots": [{{"type": "<slot type>", "value": "<text>"}}]}}

Rules:
- "intent" must be exactly one of the intents listed below.
- Each slot "type" must be exactly one of the slot types listed below.
- Each slot "value" must be copied verbatim from the request: the same words in the same order, with no rewording, no correction and no normalization.
- List the slots in the order they appear in the request. Repeat a slot type if it occurs more than once.
- If the request contains no slots, use an empty list: "slots": [].
- Do not add any other keys.

Intents: {intents}

Slot types: {slot_types}"""


@dataclass(frozen=True)
class PromptSpec:
    name: str
    k: int  # retrieved examples shown before the request
    uses_labels: bool  # does the system message list the label inventory
    description: str


PROMPTS: dict[str, PromptSpec] = {
    "finetuned_v1": PromptSpec(
        "finetuned_v1", 0, False, "One-line instruction plus the request. For the fine-tuned model."
    ),
    "zeroshot_v1": PromptSpec(
        "zeroshot_v1", 0, True, "Instructions, both label lists, and 'copy spans verbatim'."
    ),
    "fewshot_k10_v1": PromptSpec(
        "fewshot_k10_v1",
        10,
        True,
        "The zero-shot prompt plus the 10 most similar training examples as prior turns.",
    ),
}


def get_prompt(name: str) -> PromptSpec:
    try:
        return PROMPTS[name]
    except KeyError:
        raise KeyError(f"unknown prompt {name!r}; known prompts: {sorted(PROMPTS)}") from None


def static_prefix(name: str, inventory: LabelInventory | None = None) -> str:
    """The system message: the part of the prompt that is identical for every request."""
    spec = get_prompt(name)
    if not spec.uses_labels:
        return FINETUNED_INSTRUCTION
    if inventory is None:
        raise ValueError(f"prompt {name!r} lists the labels, so it needs the label inventory")
    return _ZEROSHOT_TEMPLATE.format(
        intents=", ".join(inventory.intents), slot_types=", ".join(inventory.slot_types)
    )


def render_messages(
    name: str,
    text: str,
    *,
    inventory: LabelInventory | None = None,
    shots: Sequence[Example] = (),
) -> list[dict[str, str]]:
    """The chat messages for one request.

    `shots` are the retrieved examples, MOST SIMILAR FIRST, exactly as `RetrievalIndex.topk`
    returns them. They are shown least similar first, so the closest example sits right next
    to the request. A prompt with k=0 takes no shots; one with k>0 takes exactly k.
    """
    spec = get_prompt(name)
    if len(shots) != spec.k:
        raise ValueError(f"prompt {name!r} takes exactly {spec.k} retrieved examples, got {len(shots)}")
    messages = [{"role": "system", "content": static_prefix(name, inventory)}]
    for shot in reversed(shots):
        messages.append({"role": "user", "content": shot.text})
        messages.append({"role": "assistant", "content": target_json(shot)})
    messages.append({"role": "user", "content": text})
    return messages


def finetune_record(example: Example) -> dict[str, Any]:
    """One supervised training record: the `finetuned_v1` prompt, then the labelled call."""
    messages = render_messages("finetuned_v1", example.text)
    messages.append({"role": "assistant", "content": target_json(example)})
    return {"messages": messages}


def _probe_shots(k: int, inventory: LabelInventory | None) -> list[Example]:
    intent = inventory.intents[0] if inventory else "probe_intent"
    slot_type = inventory.slot_types[0] if inventory else "probe_slot"
    return [
        Example(f"probe-{i}", "train", "probe", intent, f"probe request {i}", (Slot(slot_type, f"request {i}"),))
        for i in range(k)
    ]


def prompt_hash(name: str, inventory: LabelInventory | None = None) -> str:
    """Fingerprint of what a prompt renders: its name, k, and the messages for a fixed probe.

    Rendering a probe rather than hashing the template text means a change anywhere in
    the rendering code changes the hash too.
    """
    spec = get_prompt(name)
    probe = render_messages(
        name, "probe request", inventory=inventory, shots=_probe_shots(spec.k, inventory)
    )
    payload = json.dumps({"name": name, "k": spec.k, "probe": probe}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
