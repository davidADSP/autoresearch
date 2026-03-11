"""
One-time data preparation and runtime utilities for autoresearch translation experiments.

Usage:
    uv run prepare.py

The script:
1. Downloads the Kaggle competition data with kagglehub.
2. Builds a fixed deterministic train/validation split.
3. Trains a shared BPE tokenizer over the training split.
4. Pre-tokenizes examples for the 5-minute research loop.

Data is stored in ~/dave/.cache/autoresearch/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import rustbpe
import tiktoken
import torch
from sacrebleu.metrics import BLEU, CHRF

# ---------------------------------------------------------------------------
# Fixed experiment constants
# ---------------------------------------------------------------------------

COMPETITION_SLUG = "deep-past-initiative-machine-translation"
TIME_BUDGET = 300
MAX_SEQ_LEN = 320
MAX_PROMPT_TOKENS = 192
MAX_TARGET_TOKENS = 128
VAL_FRACTION = 0.10
VAL_MIN_EXAMPLES = 128
VAL_MAX_EXAMPLES = 512
VOCAB_SIZE = 4096

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

CACHE_DIR = Path.home() / "dave" / ".cache" / "autoresearch"
RAW_DIR = CACHE_DIR / "kaggle" / COMPETITION_SLUG
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
PROCESSED_DIR = CACHE_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.pkl"
VAL_PATH = PROCESSED_DIR / "val.pkl"
TEST_PATH = PROCESSED_DIR / "test.pkl"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|source|>",
    "<|target|>",
]
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
SOURCE_TOKEN = "<|source|>"
TARGET_TOKEN = "<|target|>"

SOURCE_COLUMN_CANDIDATES = [
    "akkadian",
    "source",
    "src",
    "input",
    "text",
    "transliteration",
    "old_assyrian",
    "old_assyrian_text",
]
TARGET_COLUMN_CANDIDATES = [
    "english",
    "target",
    "translation",
    "tgt",
    "label",
    "output",
    "reference",
]
ID_COLUMN_CANDIDATES = ["id", "ID", "row_id"]

_SPLIT_CACHE = {}
_METADATA_CACHE = None

# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------


def _load_dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def _build_kaggle_env() -> dict[str, str]:
    env = dict(os.environ)
    dotenv_values = _load_dotenv_values(Path.cwd() / ".env")
    for key, value in dotenv_values.items():
        env.setdefault(key, value)
    return env


def _has_kaggle_auth(env: dict[str, str]) -> bool:
    return bool(env.get("KAGGLE_API_TOKEN"))


def _print_kaggle_help() -> None:
    print("Kaggle authentication was not found.")
    print()
    print("Expected setup:")
    print("  1. Export KAGGLE_API_TOKEN in your shell environment, or")
    print("  2. Put KAGGLE_API_TOKEN=... in a local .env file at the repo root.")
    print()
    print("You must also accept the competition rules in your browser before downloading:")
    print(f"  https://www.kaggle.com/competitions/{COMPETITION_SLUG}")


def _extract_downloads(raw_dir: Path) -> None:
    for zip_path in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)


def _materialize_downloaded_files(search_roots: list[Path], required_names: list[str]) -> list[str]:
    found_paths: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.name in required_names:
            found_paths[root.name] = root
            continue
        if root.is_dir():
            _extract_downloads(root)
            for name in required_names:
                if name in found_paths:
                    continue
                matches = list(root.rglob(name))
                if matches:
                    found_paths[name] = matches[0]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, source_path in found_paths.items():
        destination = RAW_DIR / name
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)

    return [name for name in required_names if not (RAW_DIR / name).exists()]


def download_competition_data(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    required_names = ["train.csv", "test.csv", "sample_submission.csv"]
    required = [RAW_DIR / name for name in required_names]
    if not force and all(path.exists() for path in required):
        print(f"Data: competition files already present at {RAW_DIR}")
        return

    env = _build_kaggle_env()
    if not _has_kaggle_auth(env):
        _print_kaggle_help()
        sys.exit(1)

    os.environ["KAGGLE_API_TOKEN"] = env["KAGGLE_API_TOKEN"]
    os.environ.setdefault("KAGGLEHUB_CACHE", str(CACHE_DIR / "kagglehub"))

    try:
        import kagglehub
    except ImportError:
        print("The `kagglehub` package is not installed in this environment.")
        print("Run `uv sync` once to install the updated dependencies, then rerun `uv run prepare.py`.")
        sys.exit(1)

    print(f"Data: downloading Kaggle competition files to {RAW_DIR}")
    try:
        try:
            download_path = kagglehub.competition_download(
                COMPETITION_SLUG,
                output_dir=str(RAW_DIR),
                force_download=force,
            )
        except TypeError:
            download_path = kagglehub.competition_download(COMPETITION_SLUG, force_download=force)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Make sure your KAGGLE_API_TOKEN is valid and the competition rules were accepted.")
        sys.exit(1)

    search_roots = [RAW_DIR]
    if download_path:
        search_roots.append(Path(download_path))
    missing = _materialize_downloaded_files(search_roots, required_names)
    if missing:
        print(f"Download completed, but expected files are missing: {missing}")
        sys.exit(1)
    print("Data: download and extraction complete")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _normalize_colname(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum() or ch == "_")


def _pick_column(columns, candidates):
    normalized = {_normalize_colname(col): col for col in columns}
    for candidate in candidates:
        match = normalized.get(_normalize_colname(candidate))
        if match is not None:
            return match
    return None


def _pick_text_columns(df: pd.DataFrame) -> list[str]:
    text_cols = []
    for col in df.columns:
        series = df[col]
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            text_cols.append(col)
    return text_cols


def _detect_schema(train_df: pd.DataFrame, test_df: pd.DataFrame, sample_df: pd.DataFrame) -> dict[str, str]:
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    test_col_set = set(test_cols)
    sample_cols = list(sample_df.columns)

    id_col = _pick_column(sample_cols, ID_COLUMN_CANDIDATES) or _pick_column(train_cols, ID_COLUMN_CANDIDATES)
    submission_col = None
    if id_col and id_col in sample_cols:
        non_id = [col for col in sample_cols if col != id_col]
        if len(non_id) == 1:
            submission_col = non_id[0]
    if submission_col is None:
        submission_col = _pick_column(sample_cols, TARGET_COLUMN_CANDIDATES)

    source_col = _pick_column(train_cols, SOURCE_COLUMN_CANDIDATES)
    target_col = _pick_column(train_cols, TARGET_COLUMN_CANDIDATES)

    if target_col is None and submission_col in train_cols:
        target_col = submission_col

    if source_col is None or target_col is None:
        text_cols = [col for col in _pick_text_columns(train_df) if col != id_col]
        if source_col is None:
            shared_with_test = [col for col in text_cols if col in test_col_set]
            if shared_with_test:
                source_col = shared_with_test[0]
        if target_col is None:
            remaining = [col for col in text_cols if col != source_col]
            if submission_col in remaining:
                target_col = submission_col
            elif remaining:
                target_col = remaining[0]

    if source_col is None or target_col is None:
        raise ValueError(
            "Could not infer the Kaggle schema. "
            f"Train columns: {train_cols}, test columns: {test_cols}, sample columns: {sample_cols}"
        )

    test_source_col = source_col if source_col in test_col_set else _pick_column(test_cols, SOURCE_COLUMN_CANDIDATES)
    if test_source_col is None:
        test_text_cols = [col for col in _pick_text_columns(test_df) if col != id_col]
        if source_col in test_text_cols:
            test_source_col = source_col
        elif test_text_cols:
            test_source_col = test_text_cols[0]
    if test_source_col is None:
        raise ValueError(f"Could not infer the source column for test.csv. Test columns: {test_cols}")

    return {
        "id_col": id_col,
        "source_col": source_col,
        "target_col": target_col,
        "test_source_col": test_source_col,
        "submission_col": submission_col,
    }


def _normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).replace("\u200b", " ")
    return " ".join(text.split())


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clamped_val_size(n_rows: int) -> int:
    desired = int(round(n_rows * VAL_FRACTION))
    desired = max(desired, VAL_MIN_EXAMPLES)
    desired = min(desired, VAL_MAX_EXAMPLES)
    return max(1, min(desired, n_rows - 1))


def load_raw_frames():
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    test_df = pd.read_csv(RAW_DIR / "test.csv")
    sample_df = pd.read_csv(RAW_DIR / "sample_submission.csv")
    return train_df, test_df, sample_df


def build_fixed_split():
    train_df, test_df, sample_df = load_raw_frames()
    schema = _detect_schema(train_df, test_df, sample_df)
    id_col = schema["id_col"]
    source_col = schema["source_col"]
    target_col = schema["target_col"]

    rows = []
    for _, row in train_df.iterrows():
        source = _normalize_text(row[source_col])
        target = _normalize_text(row[target_col])
        if not source or not target:
            continue
        item = {
            "id": _normalize_text(row[id_col]) if id_col and id_col in row else "",
            "source": source,
            "target": target,
        }
        item["sort_key"] = _stable_hash(source + "\x1f" + target)
        rows.append(item)

    if len(rows) < 2:
        raise ValueError("Not enough non-empty training rows to create a train/validation split.")

    rows.sort(key=lambda item: item["sort_key"])
    n_val = _clamped_val_size(len(rows))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    for row in train_rows:
        row.pop("sort_key", None)
    for row in val_rows:
        row.pop("sort_key", None)

    test_rows = []
    for _, row in test_df.iterrows():
        source = _normalize_text(row[schema["test_source_col"]])
        if not source:
            continue
        test_rows.append(
            {
                "id": _normalize_text(row[id_col]) if id_col and id_col in row else "",
                "source": source,
            }
        )

    metadata = {
        "competition_slug": COMPETITION_SLUG,
        "schema": schema,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "test_examples": len(test_rows),
        "val_fraction": VAL_FRACTION,
        "val_cap": VAL_MAX_EXAMPLES,
        "max_seq_len": MAX_SEQ_LEN,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_target_tokens": MAX_TARGET_TOKENS,
    }
    return train_rows, val_rows, test_rows, metadata


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _train_text_iterator(train_rows):
    for row in train_rows:
        yield row["source"]
        yield row["target"]


def train_tokenizer(train_rows):
    tokenizer_pkl = TOKENIZER_DIR / "tokenizer.pkl"
    if tokenizer_pkl.exists():
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    print("Tokenizer: training shared BPE tokenizer on train split")
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(_train_text_iterator(train_rows), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(token): rank for token, rank in tokenizer.get_mergeable_ranks()}
    special_offset = len(mergeable_ranks)
    special_tokens = {name: special_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="deep-past-bpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: saved to {tokenizer_pkl}")


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)
        self.eos_token_id = enc.encode_single_token(EOS_TOKEN)
        self.pad_token_id = enc.encode_single_token(PAD_TOKEN)
        self.source_token_id = enc.encode_single_token(SOURCE_TOKEN)
        self.target_token_id = enc.encode_single_token(TARGET_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir: Path = TOKENIZER_DIR):
        with open(tokenizer_dir / "tokenizer.pkl", "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode_text(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode_text(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


# ---------------------------------------------------------------------------
# Example encoding
# ---------------------------------------------------------------------------


def _truncate(ids: list[int], max_len: int) -> list[int]:
    return ids[:max_len] if len(ids) > max_len else ids


def _encode_example(tokenizer: Tokenizer, source: str, target: str | None):
    source_ids = _truncate(tokenizer.encode_text(source), MAX_PROMPT_TOKENS - 3)
    prompt_ids = [tokenizer.bos_token_id, tokenizer.source_token_id] + source_ids + [tokenizer.target_token_id]

    if target is None:
        return {
            "prompt_ids": prompt_ids,
            "source": source,
        }

    target_ids = _truncate(tokenizer.encode_text(target), MAX_TARGET_TOKENS - 1)
    full_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    if len(full_ids) > MAX_SEQ_LEN:
        keep_target = max(1, MAX_SEQ_LEN - len(prompt_ids) - 1)
        target_ids = target_ids[:keep_target]
        full_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]

    return {
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "loss_mask_start": len(prompt_ids) - 1,
        "source": source,
        "target": target,
    }


def encode_datasets(train_rows, val_rows, test_rows):
    tokenizer = Tokenizer.from_directory()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_examples = [_encode_example(tokenizer, row["source"], row["target"]) for row in train_rows]
    val_examples = [_encode_example(tokenizer, row["source"], row["target"]) for row in val_rows]
    test_examples = [
        {
            "id": row["id"],
            **_encode_example(tokenizer, row["source"], None),
        }
        for row in test_rows
    ]

    with open(TRAIN_PATH, "wb") as f:
        pickle.dump(train_examples, f)
    with open(VAL_PATH, "wb") as f:
        pickle.dump(val_examples, f)
    with open(TEST_PATH, "wb") as f:
        pickle.dump(test_examples, f)

    print(f"Encoded train examples: {len(train_examples)}")
    print(f"Encoded val examples:   {len(val_examples)}")
    print(f"Encoded test examples:  {len(test_examples)}")


# ---------------------------------------------------------------------------
# Runtime utilities imported by train.py
# ---------------------------------------------------------------------------


def get_metadata():
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        with open(METADATA_PATH) as f:
            _METADATA_CACHE = json.load(f)
    return _METADATA_CACHE


def load_split(split: str):
    if split not in _SPLIT_CACHE:
        path = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}[split]
        with open(path, "rb") as f:
            _SPLIT_CACHE[split] = pickle.load(f)
    return _SPLIT_CACHE[split]


def make_dataloader(batch_size: int, split: str, device: str = "cuda", seed: int = 1337):
    assert split in {"train", "val"}
    examples = load_split(split)
    tokenizer = Tokenizer.from_directory()
    pad_id = tokenizer.pad_token_id

    epoch = 0
    while True:
        indices = list(range(len(examples)))
        if split == "train":
            g = torch.Generator()
            g.manual_seed(seed + epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        for start in range(0, len(indices), batch_size):
            batch = [examples[i] for i in indices[start:start + batch_size]]
            batch_size_now = len(batch)
            max_len = max(len(example["full_ids"]) - 1 for example in batch)

            x = torch.full((batch_size_now, max_len), pad_id, dtype=torch.long)
            y = torch.full((batch_size_now, max_len), -100, dtype=torch.long)

            for row_idx, example in enumerate(batch):
                full_ids = example["full_ids"]
                seq_len = len(full_ids) - 1
                x[row_idx, :seq_len] = torch.tensor(full_ids[:-1], dtype=torch.long)
                targets = torch.tensor(full_ids[1:], dtype=torch.long)
                mask_start = min(example["loss_mask_start"], seq_len)
                if mask_start > 0:
                    targets[:mask_start] = -100
                y[row_idx, :seq_len] = targets

            if device:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            yield x, y, epoch + 1

        epoch += 1
        if split == "val":
            break


def _decode_generated(tokenizer: Tokenizer, sequence: torch.Tensor, prompt_len: int) -> str:
    ids = sequence[prompt_len:].tolist()
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    ids = [token for token in ids if token not in {tokenizer.pad_token_id}]
    return tokenizer.decode_text(ids).strip()


@torch.no_grad()
def evaluate_translation(model, tokenizer: Tokenizer, batch_size: int, max_new_tokens: int = MAX_TARGET_TOKENS):
    val_examples = load_split("val")
    device = next(model.parameters()).device
    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)
    predictions = []
    references = []

    autocast_enabled = device.type == "cuda"

    for start in range(0, len(val_examples), batch_size):
        batch = val_examples[start:start + batch_size]
        prompt_lens = [len(example["prompt_ids"]) for example in batch]
        max_prompt_len = max(prompt_lens)
        input_ids = torch.full((len(batch), max_prompt_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for row_idx, example in enumerate(batch):
            prompt_ids = torch.tensor(example["prompt_ids"], dtype=torch.long, device=device)
            input_ids[row_idx, :prompt_ids.numel()] = prompt_ids

        lengths = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
        max_total_len = min(MAX_SEQ_LEN, max_prompt_len + max_new_tokens)
        row_indices = torch.arange(len(batch), device=device)

        for _ in range(max_new_tokens):
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(input_ids)
            next_token_logits = logits[row_indices, lengths - 1]
            next_tokens = next_token_logits.argmax(dim=-1)
            next_tokens = torch.where(finished, torch.full_like(next_tokens, tokenizer.pad_token_id), next_tokens)

            if input_ids.size(1) < max_total_len:
                pad_col = torch.full((len(batch), 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
                input_ids = torch.cat([input_ids, pad_col], dim=1)

            can_write = (~finished) & (lengths < input_ids.size(1))
            input_ids[row_indices[can_write], lengths[can_write]] = next_tokens[can_write]
            lengths = lengths + can_write.long()
            finished |= next_tokens.eq(tokenizer.eos_token_id)

            if finished.all() or lengths.max().item() >= max_total_len:
                break

        for row_idx, example in enumerate(batch):
            prediction = _decode_generated(tokenizer, input_ids[row_idx, :lengths[row_idx]], prompt_lens[row_idx])
            predictions.append(prediction)
            references.append(example["target"])

    bleu = bleu_metric.corpus_score(predictions, [references]).score
    chrf = chrf_metric.corpus_score(predictions, [references]).score
    score = math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))
    return {
        "score": score,
        "bleu": bleu,
        "chrf": chrf,
        "num_examples": len(references),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare Deep Past translation data for autoresearch")
    parser.add_argument("--force-download", action="store_true", help="Redownload the Kaggle competition archive")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    download_competition_data(force=args.force_download)
    print()

    train_rows, val_rows, test_rows, metadata = build_fixed_split()
    print(f"Train examples: {len(train_rows)}")
    print(f"Val examples:   {len(val_rows)}")
    print(f"Test examples:  {len(test_rows)}")
    print(f"Schema: source={metadata['schema']['source_col']}, target={metadata['schema']['target_col']}")
    print()

    train_tokenizer(train_rows)
    print()

    encode_datasets(train_rows, val_rows, test_rows)
    metadata["vocab_size"] = Tokenizer.from_directory().get_vocab_size()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print()
    print(f"Saved metadata to {METADATA_PATH}")
    print("Done! Ready to train.")


if __name__ == "__main__":
    main()
