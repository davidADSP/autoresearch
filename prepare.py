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
import re
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
PREP_VERSION = "seq2seq-mbr-v1"

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
TOKENIZER_META_PATH = TOKENIZER_DIR / "meta.json"

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
# Akkadian-specific normalization helpers
# ---------------------------------------------------------------------------

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"})

_ALLOWED_FRACS = [
    (1 / 6, "0.16666"),
    (1 / 4, "0.25"),
    (1 / 3, "0.33333"),
    (1 / 2, "0.5"),
    (2 / 3, "0.66666"),
    (3 / 4, "0.75"),
    (5 / 6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_WS_RE = re.compile(r"\s+")

_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I,
)

_CHAR_TRANS = str.maketrans(
    {
        "ḫ": "h",
        "Ḫ": "H",
        "ʾ": "",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "—": "-",
        "–": "-",
    }
)
_SUB_X = "ₓ"
_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚",
    "0.6666": "⅔",
    "0.3333": "⅓",
    "0.1666": "⅙",
    "0.625": "⅝",
    "0.75": "¾",
    "0.25": "¼",
    "0.5": "½",
}

_PN_RE = re.compile(r"\bPN\b")
_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
_CURLY_DQ_RE = re.compile("[\u201c\u201d]")
_CURLY_SQ_RE = re.compile("[\u2018\u2019]")
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")
_FORBIDDEN_TRANS = str.maketrans("", "", '——<>⌈⌋⌊[]+ʾ;')
_COMMODITY_RE = re.compile(r"(?<=\s)-(gold|tax|textiles)\b")
_COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}
_SHEKEL_REPLS = [
    (re.compile(r"5\s+11\s*/\s*12\s+shekels?", re.I), "6 shekels less 15 grains"),
    (re.compile(r"5\s*/\s*12\s+shekels?", re.I), "⅓ shekel 15 grains"),
    (re.compile(r"7\s*/\s*12\s+shekels?", re.I), "½ shekel 15 grains"),
    (re.compile(r"1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?", re.I), "15 grains"),
]
_SLASH_ALT_RE = re.compile(r"(?<![0-9/])\s+/\s+(?![0-9])\S+")
_STRAY_MARKS_RE = re.compile(r"<<[^>]*>>|<(?!gap\b)[^>]*>")
_MULTI_GAP_RE = re.compile(r"(?:<gap>\s*){2,}")
_EXTRA_STRAY_RE = re.compile(r"(?<!\w)(?:\.\.+|xx+)(?!\w)")
_HACEK_TRANS = str.maketrans({"ḫ": "h", "Ḫ": "H"})


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _ascii_to_diacritics(text: str) -> str:
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = text.replace("s,", "ṣ").replace("S,", "Ṣ")
    text = text.replace("t,", "ṭ").replace("T,", "Ṭ")
    text = _V2.sub(lambda match: match.group(1).translate(_ACUTE), text)
    text = _V3.sub(lambda match: match.group(1).translate(_GRAVE), text)
    return text


def _canon_decimal(value: float) -> str:
    integer_part = int(math.floor(value + 1e-12))
    fraction = value - integer_part
    best = min(_ALLOWED_FRACS, key=lambda pair: abs(fraction - pair[0]))
    if abs(fraction - best[0]) <= _FRAC_TOL:
        decimal = best[1]
        if integer_part == 0:
            return decimal
        return f"{integer_part}{decimal[1:]}" if decimal.startswith("0.") else f"{integer_part}+{decimal}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _frac_repl(match: re.Match) -> str:
    return _EXACT_FRAC_MAP[match.group(0)]


def _commodity_repl(match: re.Match) -> str:
    return _COMMODITY_REPL[match.group(1)]


def _month_repl(match: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(match.group(1).upper(), match.group(1))}"


def _normalize_gaps(text: str) -> str:
    return _GAP_UNIFIED_RE.sub("<gap>", text)


def _normalize_source_text(value) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    text = _ascii_to_diacritics(text)
    text = _DET_UPPER_RE.sub(r"\1", text)
    text = _DET_LOWER_RE.sub(r"{\1}", text)
    text = _normalize_gaps(text)
    text = text.translate(_CHAR_TRANS).replace(_SUB_X, "")
    text = _KUBABBAR_RE.sub("KÙ.BABBAR", text)
    text = _EXACT_FRAC_RE.sub(_frac_repl, text)
    text = _FLOAT_RE.sub(lambda match: _canon_decimal(float(match.group(1))), text)
    return _WS_RE.sub(" ", text).strip()


def postprocess_translation_text(value) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    text = _normalize_gaps(text)
    text = _PN_RE.sub("<gap>", text)
    text = _COMMODITY_RE.sub(_commodity_repl, text)
    for pattern, repl in _SHEKEL_REPLS:
        text = pattern.sub(repl, text)
    text = _EXACT_FRAC_RE.sub(_frac_repl, text)
    text = _FLOAT_RE.sub(lambda match: _canon_decimal(float(match.group(1))), text)
    text = _SOFT_GRAM_RE.sub(" ", text)
    text = _BARE_GRAM_RE.sub(" ", text)
    text = _UNCERTAIN_RE.sub("", text)
    text = _STRAY_MARKS_RE.sub("", text)
    text = _EXTRA_STRAY_RE.sub("", text)
    text = _SLASH_ALT_RE.sub("", text)
    text = _CURLY_DQ_RE.sub('"', text)
    text = _CURLY_SQ_RE.sub("'", text)
    text = _MONTH_RE.sub(_month_repl, text)
    text = _MULTI_GAP_RE.sub("<gap>", text)
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.translate(_FORBIDDEN_TRANS)
    text = text.replace("\x00GAP\x00", " <gap> ")
    text = text.translate(_HACEK_TRANS)
    text = _REPEAT_WORD_RE.sub(r"\1", text)
    for n in range(4, 1, -1):
        pattern = re.compile(r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+")
        text = pattern.sub(r"\1", text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    return _WS_RE.sub(" ", text).strip()

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
        source = _normalize_source_text(row[source_col])
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
        source = _normalize_source_text(row[schema["test_source_col"]])
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
        "prep_version": PREP_VERSION,
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
    expected_meta = {
        "prep_version": PREP_VERSION,
        "vocab_size": VOCAB_SIZE,
        "special_tokens": SPECIAL_TOKENS,
    }
    if tokenizer_pkl.exists() and _read_json_if_exists(TOKENIZER_META_PATH) == expected_meta:
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    if TOKENIZER_DIR.exists():
        shutil.rmtree(TOKENIZER_DIR)

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
    with open(TOKENIZER_META_PATH, "w") as f:
        json.dump(expected_meta, f, indent=2, sort_keys=True)
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
    encoder_ids = [tokenizer.bos_token_id, tokenizer.source_token_id] + source_ids + [tokenizer.eos_token_id]

    if target is None:
        return {
            "prompt_ids": prompt_ids,
            "source_ids": encoder_ids,
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
        "source_ids": encoder_ids,
        "decoder_input_ids": [tokenizer.target_token_id] + target_ids,
        "labels": target_ids + [tokenizer.eos_token_id],
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
            max_src_len = max(len(example["source_ids"]) for example in batch)
            max_tgt_len = max(len(example["decoder_input_ids"]) for example in batch)

            source_ids = torch.full((batch_size_now, max_src_len), pad_id, dtype=torch.long)
            source_mask = torch.zeros((batch_size_now, max_src_len), dtype=torch.bool)
            decoder_input_ids = torch.full((batch_size_now, max_tgt_len), pad_id, dtype=torch.long)
            labels = torch.full((batch_size_now, max_tgt_len), -100, dtype=torch.long)

            for row_idx, example in enumerate(batch):
                src_len = len(example["source_ids"])
                tgt_len = len(example["decoder_input_ids"])
                source_ids[row_idx, :src_len] = torch.tensor(example["source_ids"], dtype=torch.long)
                source_mask[row_idx, :src_len] = True
                decoder_input_ids[row_idx, :tgt_len] = torch.tensor(example["decoder_input_ids"], dtype=torch.long)
                labels[row_idx, :tgt_len] = torch.tensor(example["labels"], dtype=torch.long)

            if device:
                source_ids = source_ids.to(device, non_blocking=True)
                source_mask = source_mask.to(device, non_blocking=True)
                decoder_input_ids = decoder_input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            yield source_ids, source_mask, decoder_input_ids, labels, epoch + 1

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

        if hasattr(model, "generate_translations"):
            max_src_len = max(len(example["source_ids"]) for example in batch)
            source_ids = torch.full((len(batch), max_src_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            source_mask = torch.zeros((len(batch), max_src_len), dtype=torch.bool, device=device)
            for row_idx, example in enumerate(batch):
                src = torch.tensor(example["source_ids"], dtype=torch.long, device=device)
                source_ids[row_idx, :src.numel()] = src
                source_mask[row_idx, :src.numel()] = True
            batch_predictions = model.generate_translations(
                source_ids=source_ids,
                source_mask=source_mask,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            for prediction, example in zip(batch_predictions, batch):
                predictions.append(postprocess_translation_text(prediction))
                references.append(example["target"])
            continue

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
            predictions.append(postprocess_translation_text(prediction))
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


def _write_metadata(metadata: dict) -> None:
    metadata["vocab_size"] = Tokenizer.from_directory().get_vocab_size()
    metadata["prep_version"] = PREP_VERSION
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def prepare_runtime(force_download: bool = False) -> dict:
    download_competition_data(force=force_download)
    train_rows, val_rows, test_rows, metadata = build_fixed_split()
    train_tokenizer(train_rows)
    encode_datasets(train_rows, val_rows, test_rows)
    _write_metadata(metadata)
    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "metadata": metadata,
    }


def ensure_prepared_data() -> None:
    required_paths = [
        RAW_DIR / "train.csv",
        RAW_DIR / "test.csv",
        RAW_DIR / "sample_submission.csv",
        TOKENIZER_DIR / "tokenizer.pkl",
        TOKENIZER_META_PATH,
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
        METADATA_PATH,
    ]
    metadata = _read_json_if_exists(METADATA_PATH)
    tokenizer_meta = _read_json_if_exists(TOKENIZER_META_PATH)
    needs_refresh = any(not path.exists() for path in required_paths)
    needs_refresh = needs_refresh or metadata.get("prep_version") != PREP_VERSION
    needs_refresh = needs_refresh or tokenizer_meta.get("prep_version") != PREP_VERSION
    if not needs_refresh:
        return

    print("prepare.py: refreshing cached tokenizer/data for current preprocessing version")
    _SPLIT_CACHE.clear()
    global _METADATA_CACHE
    _METADATA_CACHE = None
    prepare_runtime(force_download=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare Deep Past translation data for autoresearch")
    parser.add_argument("--force-download", action="store_true", help="Redownload the Kaggle competition archive")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    prepared = prepare_runtime(force_download=args.force_download)
    train_rows = prepared["train_rows"]
    val_rows = prepared["val_rows"]
    test_rows = prepared["test_rows"]
    metadata = prepared["metadata"]
    print()

    print(f"Train examples: {len(train_rows)}")
    print(f"Val examples:   {len(val_rows)}")
    print(f"Test examples:  {len(test_rows)}")
    print(f"Schema: source={metadata['schema']['source_col']}, target={metadata['schema']['target_col']}")
    print()
    print(f"Saved metadata to {METADATA_PATH}")
    print("Done! Ready to train.")


if __name__ == "__main__":
    main()
