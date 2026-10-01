"""finetune-vs-api: a LoRA fine-tune of a small open model compared with API models.

The task is narrow on purpose: turn a free-text request into one JSON function call,
``{"intent": ..., "slots": [{"type": ..., "value": ...}]}``, using the English split of
MASSIVE 1.1 as the labelled data.
"""

__version__ = "0.1.0"
