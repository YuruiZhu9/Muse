# Teacher Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop treating `50` users as the main scaling axis, formally split `big_matrix`, expand the teacher pool from the full dataset, fine-tune the 2B Qwen student on a much larger compact SFT set, and keep a clean final offline evaluation pool.

**Architecture:** Use a time-first split. Build a large teacher candidate pool from earlier `big_matrix` blocks, sample a manageable API-labeled subset with balanced per-user coverage, fine-tune the student on compact labels, then run the student on a later time block for offline comparison. Keep teacher generation and final evaluation decoupled so the student is not trained on the same time slice used for final effectiveness claims.

**Tech Stack:** `pandas`, `jsonl`, `LLM_part/generate_teacher_from_kuairec.py`, `LLM_part/build_sft_dataset.py`, `LLM_part/SFT_Qwen1.7b/finetune_qwen4b.py`, `LLM_part/run_local_qwen_inference.py`, `run_kuairec_offline_experiment.py`

---

## Current Facts To Preserve

- `big_matrix.csv` has `12,530,806` rows, `7,176` users, `10,728` items, `28` observed dates.
- Observed time blocks are already aligned with the current teacher script:
  - `block1`: `2020-07-05` to `2020-07-12`
  - `block2`: `2020-08-01` to `2020-08-10`
  - `block3`: `2020-08-27` to `2020-09-05`
- Under the current teacher-window rule (`history < anchor_date`, `MIN_INTERACTIONS >= 10`):
  - `block1` alone can produce about `47,320` valid teacher windows across `7,001` users.
  - `block1 + block2` can produce about `118,884` valid teacher windows across `7,174` users.
- The current bottleneck is not candidate-window supply. The bottleneck is that the current generation path still centers on `top users + small max_samples`.

## Recommended Split Strategy

### Fast Validation Version

- **Teacher candidate pool:** `block1 + block2`
- **Student compact SFT train set:** sampled teacher outputs from `block1 + block2`
- **Student compact SFT dev set:** hold out `5%` to `10%` from the sampled teacher outputs
- **Final offline evaluation pool:** `block3`

This is the recommended version for the current project phase because it maximizes teacher diversity quickly and avoids time leakage into the final evaluation block.

### More Formal Version

- Keep the same time split as above.
- Additionally reserve a fixed eval-user subset from users that appear in all three blocks.
- Do not generate teacher data for those eval users.

This is better for a later report, but it is not the first thing to do if the immediate goal is to validate whether teacher expansion improves the algorithm.

## Concrete Recommendation

- Use `block1 + block2` as the teacher source.
- Do **not** use `block3` to generate teacher data if `block3` is intended for the final effect validation.
- Sample teacher data from **all eligible users**, not just top-`K` users by interaction count.
- Prefer balanced per-user sampling over naive front truncation.
- First serious teacher batch target:
  - `3,000` to `5,000` samples if API time is a concern.
  - `7,000+` samples if API throughput is acceptable.

Why this size:
- `cap=1` over the full `block1 + block2` teacher pool already gives about `7,174` balanced samples.
- That gives much higher user diversity than the current `u20/u50` path.
- It is large enough to test whether the student meaningfully improves over the current small-batch result.

---

### Task 1: Freeze The Dataset Split Definition

**Files:**
- Modify: `LLM_part/generate_teacher_from_kuairec.py`
- Create: `LLM_part/kuairec_bigmatrix_split_manifest.json`

**Step 1: Add a formal split manifest**

Define a repo-local manifest with at least:
- `teacher_blocks`
- `eval_blocks`
- optional `teacher_user_ids`
- optional `eval_user_ids`
- sampling metadata such as `min_interactions`, `max_windows_per_user`, `sample_budget`

Suggested first manifest:

```json
{
  "version": "2026-04-06-teacher-expand-v1",
  "matrix": "big",
  "teacher_blocks": ["block1", "block2"],
  "eval_blocks": ["block3"],
  "min_interactions": 10,
  "max_history_per_sample": 30
}
```

**Step 2: Stop hard-coding teacher generation to `train|val|test` only**

Add support for:
- combined teacher split such as `teacher`
- optional explicit block list such as `--blocks block1,block2`

**Step 3: Preserve the old CLI**

Keep `train|val|test` working so old artifacts remain reproducible.

### Task 2: Change Teacher Sampling From Top-Users To Full-Pool Balanced Sampling

**Files:**
- Modify: `LLM_part/generate_teacher_from_kuairec.py`

**Step 1: Add an all-user mode**

The current logic is:

```python
user_interaction_count = self.data_loader.matrix.groupby('user_id').size()
top_users = user_interaction_count.nlargest(max_users).index.tolist()
```

This must support:
- `--all_users`
- or `--max_users 0` meaning "use all eligible users"

**Step 2: Add deterministic shuffle before balanced truncation**

Current `rebalance_samples()` preserves original user order. That still biases toward the front of the list.

Add:
- `--sample_seed`
- deterministic shuffle of user order before round-robin truncation

**Step 3: Keep per-user caps**

Retain `--max_windows_per_user`, because it is the right mechanism to prevent over-concentration on heavy users.

Recommended first run:
- `teacher_blocks = block1,block2`
- `all_users = true`
- `max_windows_per_user = 1`
- `max_samples = 5000`

### Task 3: Separate Candidate-Pool Export From API Labeling

**Files:**
- Modify: `LLM_part/generate_teacher_from_kuairec.py`

**Step 1: Add a no-API candidate export mode**

Create a mode that only writes:
- `time_window_samples_*.jsonl`
- `user_contexts_*.jsonl`

without calling the API.

**Step 2: Add API labeling over an existing context file**

Large teacher generation should support:
- resume from existing partial outputs
- re-run labeling without rebuilding the candidate pool

This matters because after teacher scaling, API generation time will be much longer than the preprocessing phase.

### Task 4: Build A Proper Larger Compact SFT Dataset

**Files:**
- Modify: `LLM_part/build_sft_dataset.py`

**Step 1: Keep compact schema unchanged**

Do not change the compact output schema. The current compact 2B path is already compatible with:
- `build_sft_dataset.py`
- `finetune_qwen4b.py`
- `run_local_qwen_inference.py`
- `src/data_loader_kuairec.py`

**Step 2: Add explicit train/dev split support**

Recommended outputs:
- `trandata_qwen35_compact_teacher_v1_train.jsonl`
- `trandata_qwen35_compact_teacher_v1_dev.jsonl`

The dev split can be either:
- random `5%` to `10%`
- or sampled from later teacher windows inside `block2`

### Task 5: Fine-Tune The 2B Student On The Expanded Teacher Set

**Files:**
- Modify: `LLM_part/SFT_Qwen1.7b/finetune_qwen4b.py`

**Step 1: Keep the model at `Qwen/Qwen3.5-2B-Base`**

Do not switch back to 4B. The project has already confirmed that `2B` is the stable local path for the `4060`.

**Step 2: Clean up file naming**

The script name can stay for now, but the defaults should stop implying `50`-sample or legacy output names.

Suggested output:
- `LLM_part/SFT_Qwen1.7b/qwen35_2b_finetuned_teacher_v1`

**Step 3: Keep LoRA and 4-bit training**

Do not introduce a larger training change until the teacher-expansion effect itself is verified.

### Task 6: Generate Final Evaluation Contexts From `block3`

**Files:**
- Modify: `LLM_part/generate_teacher_from_kuairec.py`
- Or create: `LLM_part/export_eval_contexts_from_kuairec.py`

**Step 1: Export contexts from `block3` without teacher API calls**

These contexts are for:
- running the local student
- producing semantic outputs
- feeding those outputs into embedding and ranking

**Step 2: Prefer one latest eligible context per user for the first serious experiment**

This avoids duplicate-user collapse in the current offline experiment loader and keeps evaluation accounting clean.

### Task 7: Run The Same Three-Way Offline Comparison

**Files:**
- Reuse: `run_kuairec_offline_experiment.py`
- Reuse: `src/evaluation.py`

**Step 1: Keep the comparison set fixed**

Run:
- `din_only`
- `llm_recall_only`
- `llm_full`

**Step 2: Keep `anchor_date`-aligned evaluation**

Do not go back to generic `last_positive` cutoff. The current project memory already confirms that the all-zero results were caused by temporal misalignment.

---

## Recommended Execution Order

1. Add formal split support for `block1 + block2 -> teacher`, `block3 -> eval`.
2. Change teacher sampling to all-user balanced sampling.
3. Export teacher candidate contexts without API.
4. Run API labeling for `3,000` to `5,000` teacher samples.
5. Build compact train/dev SFT files.
6. Fine-tune `Qwen3.5-2B`.
7. Export `block3` eval contexts.
8. Run local student inference on `block3`.
9. Run the three-way offline comparison.

## Commands To Target After Implementation

Teacher candidate export:

```bash
python LLM_part/generate_teacher_from_kuairec.py \
  --matrix big \
  --blocks block1,block2 \
  --all_users \
  --max_windows_per_user 1 \
  --max_samples 5000 \
  --output_tag teacher_v1 \
  --contexts_only
```

Teacher API labeling:

```bash
python LLM_part/generate_teacher_from_kuairec.py \
  --matrix big \
  --blocks block1,block2 \
  --all_users \
  --max_windows_per_user 1 \
  --max_samples 5000 \
  --output_tag teacher_v1 \
  --resume
```

Build compact SFT:

```bash
set SFT_CONTEXT_FILE=LLM_part/user_contexts_big_teacher_v1.jsonl
set SFT_INFERENCE_FILE=LLM_part/user_inferences_big_teacher_v1.jsonl
set SFT_OUTPUT_FILE=LLM_part/SFT_Qwen1.7b/trandata_qwen35_compact_teacher_v1_train.jsonl
python LLM_part/build_sft_dataset.py
```

Fine-tune:

```bash
set QWEN_SFT_DATA_PATH=LLM_part/SFT_Qwen1.7b/trandata_qwen35_compact_teacher_v1_train.jsonl
set QWEN_OUTPUT_DIR=LLM_part/SFT_Qwen1.7b/qwen35_2b_finetuned_teacher_v1
python LLM_part/SFT_Qwen1.7b/finetune_qwen4b.py
```

## Decision

For the next round, the project should **not** keep scaling by "more users in the same tiny batch".  
It should scale by:
- formalizing the `big_matrix` time split
- using `block1 + block2` as the teacher source
- sampling teacher windows from the full eligible user pool
- fine-tuning the existing stable `2B` student on a much larger compact teacher set
- keeping `block3` clean for final comparison

