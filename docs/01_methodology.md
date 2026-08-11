# Methodology

This mirrors the thesis Methodology chapter (see the delivered `Methodology_Chapter.docx`); kept here
in plain markdown as the working reference.

## Models

| Model | Params | Type | Role |
|---|---|---|---|
| Gemma-2-9B-it | 9B | Multilingual instruct | Primary testbed, all methods |
| Llama-3.1-8B-Instruct | 8B | English-primary instruct | Base model; Dorna control |
| Dorna2-Llama3.1-8B-Instruct | 8B | Persian fine-tune of Llama-3.1 | Persian-specialized model |
| BGE-m3 | — | Multilingual dense encoder | Best encoder baseline; retriever for RAG methods |
| TF-IDF, BM25 | — | Lexical | Baselines |
| Tooka-SBERT, Persian-Emb | — | Persian sentence encoders | Baselines |
| LaBSE | — | Multilingual encoder | Baseline |

All LLMs loaded 4-bit (NF4, double quant, float16 compute) to fit a T4 GPU (Colab free tier).
Global seed = 42. Greedy/deterministic decoding throughout.

## Evaluation protocol: Permutation Self-Consistency (PSC)

Per instance: generate k=5 random permutations of the 5 candidates → rank each independently → map
back to original candidate identities → aggregate by Borda count (lower total rank = better) →
final ranking. Winner-consistency = fraction of the k rankings whose top pick agrees. Invalid/unparseable
generations get a uniform max-rank penalty (never silently inflate a candidate).

## Methods, in the order explored

1. **Lean systematic prompt** — one system instruction (ranking role + direct-answer-beats-clarifying-
   question rule + strict output format), 2 turn-structured few-shot examples, no vocabulary glossary
   (an earlier heavy prompt with 5 examples + glossary caused output collapse onto a fixed ranking).
   Language-per-model fairness rule: English for Gemma/Llama, Persian for Dorna (matches each model to
   its best-performing language; not a confound).

2. **Chain-of-Prompting (CoP)** — XLT-style (reason in English over Persian) + Plan-and-Solve checklist.
   **Negative result**: no gain over lean prompt, 4–5x cost. Truncating generation to 90 tokens hurt
   results; 200 tokens fixed it (truncated CoT can be worse than none).

3. **Self-distilled curriculum LoRA (SFT-only)** — distill CoP's own correct-only traces, curriculum
   order random→domain→domain_lexical. LoRA: r=16, alpha=32, dropout=0.05, target
   q/k/v/o_proj, fp32 cast on trainable params, fp16=False, batch=1, grad_accum=8, lr=1e-4,
   max_length=1024. **Result: format validity 100%, ranking flat (0.48)** — self-distillation on
   correct-only traces never sees failure cases (exposure bias). Motivates the merge.

4. **RAG-ICL (frozen model + retrieval)** — BGE-m3 cosine-kNN retrieval of top-k solved same-strategy
   neighbours, injected as dynamic few-shot. k=3 chosen empirically (k=1/2/3 tested on n=20 subset;
   apparent k=3 gain did not fully survive to n=50 — a caution about small-sample method selection).
   **Result: Hits@1 unchanged (~0.52), winner-consistency jumped substantially** — retrieval stabilizes
   a frozen model without improving its top-1 judgment.

5. **The merge — RAG-augmented SFT (retrieval-conditioned fine-tuning)** — RA-DIT style: each training
   example prepended with its 2 nearest solved same-strategy neighbours; model trained to condition on
   retrieved context, not just see it. Fresh LoRA (not continued from SFT-only). Same LoRA hyperparams
   as above but max_length=1536 (longer, retrieved context), k=2 neighbours matched at train and
   inference time (critical — evaluating without retrieval at inference would be out-of-distribution).
   **This is the only method that broke the ~0.52 ceiling, on all three models.**

## Fair-conditions cross-model baseline

Identical protocol (listwise + PSC k=5 + Borda), each model in its best language/template
(Gemma/Llama English, Dorna Persian). This — not the mixed zero-shot numbers — is the valid basis for
cross-model comparison (see `docs/00_project_overview.md`, "Persian-model question").

## Compute environment

Google Colab free tier, single T4 GPU (16GB), no persistent compute between sessions. Google Drive
free tier (15GB) — can't hold a full model (~17GB) but LoRA adapters (tens of MB) fit fine; base
weights re-downloaded from Hugging Face each session. See `docs/04_engineering_lessons.md` for the
full list of gotchas encountered running this pipeline.
