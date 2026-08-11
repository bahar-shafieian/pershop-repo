# Engineering Lessons (Colab / T4 / Drive)

Keep this file updated — these are exactly the mistakes that cost hours before, and will again.

## Storage and persistence

- **Google Drive persists across sessions; `/content/` does not.** Anything in local Colab storage is
  wiped on disconnect/reconnect. Datasets, adapters, and any file you need next session must be on
  Drive, not just `/content/`.
- **Drive free tier (15GB) cannot hold a full model** (~17GB) but LoRA adapters (tens of MB) fit fine.
  Re-download base model weights from Hugging Face fresh each session; only save/load adapters from
  Drive.
- **Save adapters immediately after training** (`model.save_pretrained`) — they live only in GPU memory
  until saved. Lost ~3h twice by zipping the wrong folder (CSVs instead of the adapter). Bake the
  Drive-save into the training cell itself, don't do it as an afterthought.
- **Checkpoint long evals every ~30 instances to Drive**, resumable, because free-tier Colab preempts
  long runs after heavy all-day use. This is what let the Gemma merge re-eval survive a mid-run
  disconnect this session without losing progress.

## Session-disconnect recovery (this session's actual experience)

When a session disconnects mid-analysis:
1. Everything in Python memory (`hit1_lookup`, dataframes, etc.) is gone — must re-run from Drive.
2. Drive-mounted files themselves are safe — no need to re-upload adapters/results.
3. Fastest recovery: consolidate the "load from Drive → rebuild state" steps into a single cell rather
   than re-running many small cells with paste-and-check in between. See
   `notebooks/colab_cells_reference.py` for the consolidated recovery cell actually used.

## Per-instance result files are NOT interchangeable schemas

Real gotcha hit this session: three "per-instance results" files, written at different times by
different notebook versions, used three different schemas:

| File | ID field | Correctness field |
|---|---|---|
| Gemma merge (written this session) | `id` (string, e.g. `d4_t0000`) | `hit1` (0/1, direct) |
| Llama merge (original notebook) | `idx` (positional int, 0-based) | `gold_rank` (1 = correct) |
| Dorna merge (original notebook) | `idx` (positional int, 0-based) | `gold_rank` (1 = correct) |

**Always print one sample line from each file before writing a loader that assumes a shared schema.**
Positional (`idx`) alignment across files is only valid if every file evaluated the *same* test file in
the *same*, unshuffled order — worth stating as an assumption, not a guarantee, and worth a sanity check
(compare resulting Hits@1 to already-known values) before trusting the alignment.

## Model / training

- **T4 numerics**: `fp16=False` + cast trainable LoRA params to `float32` (avoids a bf16 grad-scaler
  error). `max_length=1536` needed for RAG-augmented prompts (retrieved context lengthens inputs
  considerably vs. plain prompts at `max_length=1024`).
- **TRL version drift**: `SFTConfig` uses `max_length` (not `max_seq_length`); `DPOConfig` rejects
  `max_prompt_length`; `precompute_ref_log_probs` has a `/tmp` cache bug — fix by removing/renaming the
  flagged argument.
- **Chat Vector / loading multiple full models**: don't load 3 full 8B models into RAM simultaneously —
  stream tensor-by-tensor from disk shards, or you'll OOM.
- **Merge evaluation must match train-time retrieval conditions**: evaluating a retrieval-conditioned
  adapter on bare (non-retrieval) prompts tests it out-of-distribution and understates performance.
  Always evaluate with the same retrieval setup used in training.

## Prompt design

- **Heavy prompts (many worked examples + a vocabulary glossary) can cause output collapse** — the
  model copies the literal structure/ordering from the examples rather than reasoning about the actual
  candidates. Fix: fewer examples (2 worked, turn-structured), no glossary, explicit "this is a new,
  independent instance" framing.
- **Truncated chain-of-thought can be worse than no CoT at all.** An early CoP run capped generation at
  90 tokens, truncating the reasoning checklist mid-sequence and measurably hurting results; raising to
  200 tokens fixed it.
- **Self-distillation prompts should describe the output format abstractly, not with a literal example
  ranking** — an early version embedded a concrete example ordering in the format instruction and the
  model copied it verbatim instead of reasoning; abstracting the instruction raised the correct-trace
  keep rate from 33% to 71%.

## Dataset provenance

- The PerSHOP train/test JSONL files used throughout this project are **not on Hugging Face** under any
  obvious dataset name (`HamedTaherkhani/pershop` does not exist) — they must be sourced from wherever
  they were originally obtained (uploaded directly as local files) and re-uploaded fresh each session,
  or moved to Drive once, permanently, to stop re-uploading them. **Action item: move the 6 canonical
  JSONL files (3 strategies × train/test) to a permanent Drive folder now, so this stops being a
  recurring problem.**
