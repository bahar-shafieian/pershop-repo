"""
PerSHOP benchmark loader.

Schema (one JSON object per line, JSONL):
{
  "id": "d4_t0000",
  "dialogue_number": 4,
  "domain": "...",
  "super_product": "...",
  "query": "...",                     # Persian customer message
  "query_intents": "...",
  "positive_response": "...",         # gold response text
  "response_intents": "...",
  "candidates": [
      {"text": "...", "label": 0 or 1},   # exactly one label==1 per instance
      ...  (5 total)
  ],
  "negative_strategy": "random" | "domain" | "domain_lexical",
  "split": "train" | "test"
}

NOTE ON PROVENANCE: these files are not hosted on Hugging Face under any known dataset name.
Source them from wherever they were originally obtained and keep a permanent copy on Drive —
see docs/04_engineering_lessons.md, "Dataset provenance".
"""
import json
from pathlib import Path
from typing import Iterator


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a PerSHOP JSONL file (one JSON object per line) into a list of dicts."""
    instances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def gold_index(instance: dict) -> int:
    """Return the 0-based index of the gold (label==1) candidate."""
    for i, c in enumerate(instance["candidates"]):
        if c["label"] == 1:
            return i
    raise ValueError(f"No gold candidate found in instance {instance.get('id')}")


def iter_strategy_split(data_dir: str | Path, strategy: str, split: str) -> Iterator[dict]:
    """
    Convenience iterator: yields instances for a given strategy/split.
    strategy in {"random", "domain", "domain_lexical"}, split in {"train", "test"}.
    Expects files named pershop_{strategy}_{split}.jsonl in data_dir.
    """
    path = Path(data_dir) / f"pershop_{strategy}_{split}.jsonl"
    for inst in load_jsonl(path):
        yield inst


def gold_position_balance(instances: list[dict]) -> dict[int, int]:
    """
    Diagnostic used in the thesis to rule out a data artifact behind position bias:
    counts how many instances have the gold candidate in each of the 5 slots.
    Should be roughly balanced (~55-79 per slot on domain_lexical, n=330) if the
    dataset construction is unbiased with respect to gold position.
    """
    counts = {i: 0 for i in range(5)}
    for inst in instances:
        counts[gold_index(inst)] += 1
    return counts


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pershop_loader.py <path_to_jsonl>")
        sys.exit(1)
    data = load_jsonl(sys.argv[1])
    print(f"Loaded {len(data)} instances")
    print("Example keys:", list(data[0].keys()))
    print("Gold position balance:", gold_position_balance(data))
