# autoresearch

This repo is set up for autonomous research on the Kaggle competition `deep-past-initiative-machine-translation`.

The task is to translate Akkadian into English and improve the local validation score that approximates the Kaggle leaderboard.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag: propose a tag based on today's date, such as `mar11`. The branch `autoresearch/<tag>` must not already exist.
2. Create the branch: `git checkout -b autoresearch/<tag>` from current `master`.
3. Read the in-scope files for full context:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. Verify prepared data exists under `~/dave/.cache/autoresearch/`. If not, run `uv run prepare.py`.
5. If `prepare.py` reports missing Kaggle auth, tell the human to either:
   - put `KAGGLE_API_TOKEN=...` in the local `.env`, or
   - export `KAGGLE_API_TOKEN` in the shell, or
   - use `~/.kaggle/kaggle.json`
6. Remind the human that they must accept the Kaggle competition rules in a browser before the CLI can download the files.
7. Initialize `results.tsv` with the header row only.
8. Confirm setup looks good.

Once setup is confirmed, begin the experiment loop.

## Fixed Benchmark

`prepare.py` defines the fixed local benchmark:

- It downloads the Kaggle competition files.
- It builds a deterministic train/validation split from `train.csv`.
- It trains a shared tokenizer on the training split only.
- It evaluates using a local approximation of the official metric.

Do not modify `prepare.py` during normal experiment iteration. Treat it as the fixed harness once setup is complete.

## Objective

The objective is to maximize `val_score`.

The local validation score is:

`val_score = sqrt(BLEU * chrF++)`

using a fixed deterministic validation split and `sacrebleu` metrics. Higher is better.

## Experimentation

Each experiment runs on a single NVIDIA A100 GPU. The training script runs for a fixed 5-minute wall-clock training budget:

`uv run train.py`

What you CAN do:

- Modify `train.py`
- Change the model architecture
- Change the optimizer
- Change batch size, depth, width, attention, decoding strategy, and training schedule

What you CANNOT do:

- Change the local evaluation metric in `prepare.py`
- Change the deterministic validation split in `prepare.py`
- Install extra packages beyond those already declared in `pyproject.toml`

## Simplicity Criterion

All else equal, prefer simpler changes.

- Keep changes that improve `val_score` without making the code much uglier.
- Strongly prefer deletions or simplifications that preserve score.
- Be skeptical of tiny gains that add a lot of brittle complexity.

## First Run

The first run should always establish the baseline:

1. Commit the untouched baseline state for this branch.
2. Run `uv run train.py > run.log 2>&1`
3. Read the key metrics from `run.log`
4. Record the result in `results.tsv`

## Output Format

At the end of a run, `train.py` prints:

```text
---
val_score:        37.123456
val_bleu:         31.234567
val_chrf:         44.123456
val_examples:     512
training_seconds: 300.0
startup_seconds:  5.3
total_seconds:    326.4
peak_vram_mb:     11842.7
target_tokens_M:  7.80
num_steps:        942
num_params_M:     18.70
```

Extract the key results with:

`rg "^val_score:|^val_bleu:|^val_chrf:|^peak_vram_mb:" run.log`

If the run crashes, inspect the end of the log and fix obvious issues before trying again.

## Logging Results

Log every experiment in `results.tsv` as tab-separated values with the header:

```text
commit	val_score	memory_gb	status	description
```

Columns:

1. Short git commit hash
2. `val_score` with 6 decimals, or `0.000000` for crashes
3. Peak memory in GB with 1 decimal, or `0.0` for crashes
4. `keep`, `discard`, or `crash`
5. Short description of the idea

Example:

```text
commit	val_score	memory_gb	status	description
a1b2c3d	34.512300	11.6	keep	baseline decoder-only translation model
b2c3d4e	35.104900	12.1	keep	increase model width and LR warmup
c3d4e5f	34.220100	11.5	discard	switch MLP activation
d4e5f6g	0.000000	0.0	crash	double depth caused OOM
```

Do not commit `results.tsv`.

## Experiment Loop

LOOP FOREVER:

1. Check the current git state.
2. Propose one concrete idea to improve `val_score`.
3. Edit `train.py`.
4. Commit the change.
5. Run `uv run train.py > run.log 2>&1`.
6. Read the metrics from `run.log`.
7. If the run crashed, inspect the log, decide whether the issue is worth a quick fix, and either retry or mark it as `crash`.
8. Record the result in `results.tsv`.
9. If `val_score` improved, keep the commit and continue from there.
10. If `val_score` did not improve, revert only your experiment commit and continue from the previous best point.

## Timeouts And Failures

- A run should take about 5 minutes of training plus startup and evaluation overhead.
- If a run exceeds 10 minutes total, kill it and treat it as a failed experiment.
- If a change causes obvious instability, NaNs, or OOM, move on quickly unless the fix is trivial.

## Autonomy

Once the experiment loop begins, do not stop to ask the human for permission to continue. Keep iterating until the human interrupts you.
