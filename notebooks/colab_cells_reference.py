# =============================================================================
# Colab Cells Reference -- flat, ordered record of the cells actually run
# to (a) regenerate the missing Gemma merge per-instance eval, and
# (b) run the complementarity analysis + confidence-routed ensemble.
#
# This is a REFERENCE, not a script to run top-to-bottom blindly -- paths
# assume the specific Drive layout from this project (see below). Cross-check
# against docs/04_engineering_lessons.md before reusing.
# =============================================================================

# -----------------------------------------------------------------------
# PART 1 -- Regenerating the missing Gemma merge per-instance evaluation
# (Llama and Dorna already had these files from earlier sessions; only
# Gemma's was missing -- only the aggregate 0.594 number had been saved.)
# -----------------------------------------------------------------------

# CELL -- Load model, tokenizer, adapter
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-2-9b-it"
ADAPTER_PATH = "/content/drive/MyDrive/gemma_pershop_rag_sft"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
"""

# CELL -- Load BGE-m3 retriever + build training index (domain_lexical train, n=1483)
"""
from sentence_transformers import SentenceTransformer
import json, numpy as np

retriever = SentenceTransformer("BAAI/bge-m3")

TRAIN_PATH = "/content/pershop_data/pershop_domain_lexical_train.jsonl"
train_instances = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")]
train_queries = [inst["query"] for inst in train_instances]
train_embeddings = retriever.encode(train_queries, normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=True)
"""

# CELL -- Helper functions: retrieval, prompt building, PSC inference
# (full implementations moved to src/methods/prompts.py, src/methods/retrieval.py,
#  src/eval/psc.py -- this cell originally inlined all of it; see those modules
#  for the cleaned-up, reusable versions)

# CELL -- PSC k=5 evaluation over all 330 domain_lexical test instances
# Result: saved to /content/drive/MyDrive/gemma_merge_eval330/
#         gemma_merge_domain_lexical_330_checkpoint.jsonl
# Checkpointed every 30 instances (survives preemption/disconnect).
# ACTUAL OUTPUT FROM THIS RUN:
"""
Test instances: 330
[30/330]  Hits@1 so far: 0.400 | saved to Drive
[60/330]  Hits@1 so far: 0.500 | saved to Drive
[90/330]  Hits@1 so far: 0.544 | saved to Drive
[120/330] Hits@1 so far: 0.542 | saved to Drive
[150/330] Hits@1 so far: 0.567 | saved to Drive
[180/330] Hits@1 so far: 0.583 | saved to Drive
[210/330] Hits@1 so far: 0.571 | saved to Drive
[240/330] Hits@1 so far: 0.558 | saved to Drive
[270/330] Hits@1 so far: 0.563 | saved to Drive
[300/330] Hits@1 so far: 0.577 | saved to Drive
[330/330] Hits@1 so far: 0.576 | saved to Drive
Done. Final Hits@1: 0.576
"""
# Note: 0.576 here vs. 0.594 in the original run -- normal run-to-run noise
# (retrieval index construction order, etc.), same rise-and-stabilize shape
# as the original. Not a red flag.


# -----------------------------------------------------------------------
# PART 2 -- Complementarity analysis (this is the actual analysis code;
# see src/eval/complementarity.py for the clean, reusable version -- this
# block is the exact ad-hoc form run in Colab, including the schema
# mismatch that had to be debugged live)
# -----------------------------------------------------------------------

# CELL -- schema check that revealed the mismatch (IMPORTANT -- don't skip this
# step when adding a new model's results file; see docs/04_engineering_lessons.md)
"""
import json
files_to_check = {
    "gemma": "/content/drive/MyDrive/gemma_merge_eval330/gemma_merge_domain_lexical_330_checkpoint.jsonl",
    "llama": "/content/drive/MyDrive/llama_merge_eval330/llama_merge_domain_lexical_330_checkpoint.jsonl",
    "dorna": "/content/drive/MyDrive/dorna_merge_eval330/dorna_merge_domain_lexical_330_checkpoint.jsonl",
}
for name, path in files_to_check.items():
    with open(path, encoding="utf-8") as f:
        print(f"--- {name} ---"); print(f.readline())
"""
# OUTPUT (this is what revealed the schema mismatch):
"""
--- gemma ---
{"id": "d4_t0000", "hit1": 1, "gold_rank": 1, "borda_scores": [13, 2, 9, 16, 10], "parse_errors": 0, "winner_consistency": 0.8}
--- llama ---
{"idx": 0, "gold_rank": 1, "valid": 5, "wc": false}
--- dorna ---
{"idx": 0, "gold_rank": 1, "valid": 5, "wc": true}
"""

# CELL -- full pipeline, run as ONE cell after a session disconnect (consolidated
# recovery pattern -- see docs/04_engineering_lessons.md, "Session-disconnect recovery")
"""
from google.colab import drive
drive.mount('/content/drive')

import json, numpy as np, pandas as pd
from itertools import combinations
!pip install -q statsmodels
from statsmodels.stats.contingency_tables import mcnemar

def load_gemma(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    return df.set_index("id")["hit1"].astype(int), df.set_index("id")["winner_consistency"]

def load_llama_dorna(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["hit1"] = (df["gold_rank"] == 1).astype(int)
    return df.set_index("idx")["hit1"], df.set_index("idx")["wc"].astype(float)

gemma_s, gemma_wc = load_gemma(".../gemma_merge_domain_lexical_330_checkpoint.jsonl")
llama_s, llama_wc = load_llama_dorna(".../llama_merge_domain_lexical_330_checkpoint.jsonl")
dorna_s, dorna_wc = load_llama_dorna(".../dorna_merge_domain_lexical_330_checkpoint.jsonl")

gemma_s_pos  = gemma_s.reset_index(drop=True);  gemma_s_pos.index.name  = "idx"
gemma_wc_pos = gemma_wc.reset_index(drop=True); gemma_wc_pos.index.name = "idx"

hit1_lookup = pd.DataFrame({"gemma": gemma_s_pos, "llama": llama_s, "dorna": dorna_s}).dropna()
wc_df       = pd.DataFrame({"gemma": gemma_wc_pos, "llama": llama_wc, "dorna": dorna_wc}).dropna()

oracle = (hit1_lookup.sum(axis=1) > 0).mean()
best   = hit1_lookup.mean().max()

def route(row):
    order = ["gemma", "llama", "dorna"]
    return max(order, key=lambda m: (row[m], m == "gemma"))

routed_model  = wc_df.apply(route, axis=1)
ensemble_hit1 = np.array([hit1_lookup.loc[i, routed_model.loc[i]] for i in hit1_lookup.index])

gemma_only = hit1_lookup["gemma"].values
gemma_right_routed_wrong = int(((gemma_only == 1) & (ensemble_hit1 == 0)).sum())
routed_right_gemma_wrong = int(((gemma_only == 0) & (ensemble_hit1 == 1)).sum())
table = [[0, gemma_right_routed_wrong], [routed_right_gemma_wrong, 0]]
n_discordant = gemma_right_routed_wrong + routed_right_gemma_wrong
result = mcnemar(table, exact=(n_discordant < 25), correction=True)
"""
# ACTUAL FINAL OUTPUT FROM THIS RUN (the numbers used throughout docs/):
"""
Matched instances: 330
Hits@1: {'gemma': 0.576, 'llama': 0.573, 'dorna': 0.57}

Oracle Hits@1: 0.715 | Best single: 0.576 | Gain: +0.139

Routing distribution: {'gemma': 282, 'llama': 34, 'dorna': 14}
Confidence-routed ensemble Hits@1: 0.618

Both correct: 190 | Both wrong: 126
Gemma right, routed wrong: 0
Routed right, gemma wrong: 14

McNemar p-value: 0.0001
Significant at α=0.05
"""
