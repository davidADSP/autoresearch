"""
Prepare Kaggle notebook assets for submitting the current best autoresearch model.

Kaggle code competitions do not allow direct local submission: the final commit and
submit action must happen from a Kaggle Notebook. This helper packages the current
best local checkpoint into:

1. A Kaggle Dataset folder containing model artifacts.
2. A Kaggle Notebook folder that loads those artifacts, writes `submission.csv`,
   and is configured for the Deep Past competition.

Usage:
    uv run submit_best.py
    uv run submit_best.py --dataset-slug yourname/autoresearch-deep-past-best \
        --kernel-slug yourname/autoresearch-deep-past-submit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
from pathlib import Path

COMPETITION_SLUG = "deep-past-initiative-machine-translation"
RESULTS_DIR = Path("results")
BEST_MODEL_DIR = RESULTS_DIR / "best_model"
KAGGLE_OUTPUT_DIR = RESULTS_DIR / "kaggle_submission"
CHECKPOINT_NAME = "checkpoint.pt"
TOKENIZER_NAME = "tokenizer.pkl"
METRICS_NAME = "metrics.json"

DEFAULT_DATASET_SLUG = "your-kaggle-username/autoresearch-deep-past-best"
DEFAULT_KERNEL_SLUG = "your-kaggle-username/autoresearch-deep-past-submit"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get_kaggle_username() -> str | None:
    username = os.environ.get("KAGGLE_USERNAME")
    if username:
        return username.strip()
    return _load_dotenv_values(Path(".env")).get("KAGGLE_USERNAME")


def _resolve_owner_slug(value: str | None, default_name: str) -> str:
    if value:
        return value
    username = _get_kaggle_username()
    if not username:
        raise SystemExit(
            "KAGGLE_USERNAME was not found in the environment or local .env. "
            f"Either set it or pass an explicit slug like --dataset-slug yourname/{default_name}."
        )
    return f"{username}/{default_name}"


def _dataset_mount_name(dataset_slug: str) -> str:
    return dataset_slug.split("/", 1)[-1]


def _build_notebook_code(dataset_slug: str, bundle_metrics: dict) -> str:
    dataset_mount = _dataset_mount_name(dataset_slug)
    max_seq_len = bundle_metrics.get("max_seq_len", 320)
    max_target_tokens = bundle_metrics.get("max_target_tokens", 128)

    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import pickle
        from dataclasses import dataclass
        from pathlib import Path

        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        try:
            import tiktoken  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "This notebook expects `tiktoken` to be available in the Kaggle runtime. "
                "If it is missing, attach an offline wheel or install it in a custom image first."
            ) from exc

        COMPETITION_SLUG = "{COMPETITION_SLUG}"
        EXPECTED_DATASET_DIR = Path("/kaggle/input/{dataset_mount}")
        OUTPUT_PATH = Path("/kaggle/working/submission.csv")
        MAX_SEQ_LEN = {max_seq_len}
        MAX_TARGET_TOKENS = {max_target_tokens}

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


        def _resolve_bundle_paths() -> tuple[Path, Path, Path, Path]:
            preferred = EXPECTED_DATASET_DIR
            required_names = {{"{CHECKPOINT_NAME}", "{TOKENIZER_NAME}", "{METRICS_NAME}"}}
            if preferred.exists():
                existing = {{path.name for path in preferred.iterdir() if path.is_file()}}
                if required_names.issubset(existing):
                    return (
                        preferred,
                        preferred / "{CHECKPOINT_NAME}",
                        preferred / "{TOKENIZER_NAME}",
                        preferred / "{METRICS_NAME}",
                    )

            candidates = []
            for checkpoint_path in Path("/kaggle/input").rglob("{CHECKPOINT_NAME}"):
                candidate_dir = checkpoint_path.parent
                tokenizer_path = candidate_dir / "{TOKENIZER_NAME}"
                metrics_path = candidate_dir / "{METRICS_NAME}"
                if tokenizer_path.exists() and metrics_path.exists():
                    candidates.append((candidate_dir, checkpoint_path, tokenizer_path, metrics_path))

            if not candidates:
                raise FileNotFoundError(
                    "Could not find the exported model bundle under /kaggle/input. "
                    f"Expected files: {{sorted(required_names)}}. "
                    "Make sure the exported dataset is attached to the notebook version."
                )

            dataset_dir, checkpoint_path, tokenizer_path, metrics_path = candidates[0]
            print(f"Using model bundle from {{dataset_dir}}")
            return dataset_dir, checkpoint_path, tokenizer_path, metrics_path


        def _resolve_competition_paths() -> tuple[Path, Path]:
            candidates = []
            for test_path in Path("/kaggle/input").rglob("test.csv"):
                parent = test_path.parent
                sample_path = parent / "sample_submission.csv"
                if sample_path.exists():
                    candidates.append((test_path, sample_path))

            if not candidates:
                raise FileNotFoundError(
                    "Could not find competition files under /kaggle/input. "
                    "Attach the competition dataset to the notebook version."
                )

            preferred = []
            fallback = []
            for test_path, sample_path in candidates:
                path_str = str(test_path.parent).lower()
                if "deep-past" in path_str or "translation" in path_str:
                    preferred.append((test_path, sample_path))
                else:
                    fallback.append((test_path, sample_path))

            test_path, sample_path = (preferred or fallback)[0]
            print(f"Using competition data from {{test_path.parent}}")
            return test_path, sample_path


        DATASET_DIR, CHECKPOINT_PATH, TOKENIZER_PATH, METRICS_PATH = _resolve_bundle_paths()
        TEST_PATH, SAMPLE_PATH = _resolve_competition_paths()


        def _normalize_colname(name: str) -> str:
            return "".join(ch.lower() for ch in str(name) if ch.isalnum() or ch == "_")


        def _pick_column(columns, candidates):
            normalized = {{_normalize_colname(col): col for col in columns}}
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


        def _detect_schema(test_df: pd.DataFrame, sample_df: pd.DataFrame) -> dict[str, str]:
            test_cols = list(test_df.columns)
            sample_cols = list(sample_df.columns)
            test_col_set = set(test_cols)

            id_col = _pick_column(sample_cols, ID_COLUMN_CANDIDATES) or _pick_column(test_cols, ID_COLUMN_CANDIDATES)
            submission_col = None
            if id_col and id_col in sample_cols:
                non_id = [col for col in sample_cols if col != id_col]
                if len(non_id) == 1:
                    submission_col = non_id[0]
            if submission_col is None:
                submission_col = _pick_column(sample_cols, TARGET_COLUMN_CANDIDATES)

            source_col = _pick_column(test_cols, SOURCE_COLUMN_CANDIDATES)
            if source_col is None:
                test_text_cols = [col for col in _pick_text_columns(test_df) if col != id_col]
                if test_text_cols:
                    source_col = test_text_cols[0]

            if source_col is None or submission_col is None:
                raise ValueError(
                    "Could not infer Kaggle schema. "
                    f"Test columns: {{test_cols}}, sample columns: {{sample_cols}}"
                )

            if source_col not in test_col_set:
                raise ValueError(f"Detected source column '{{source_col}}' is not present in test.csv")

            return {{
                "id_col": id_col,
                "source_col": source_col,
                "submission_col": submission_col,
            }}


        def _normalize_text(value) -> str:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            text = str(value).replace("\\u200b", " ")
            return " ".join(text.split())


        def _normalize_prediction_text(value) -> str:
            text = _normalize_text(value)
            # Kaggle submissions must not contain null/blank prediction cells.
            return text if text else "[blank]"


        class Tokenizer:
            def __init__(self, tokenizer_path: Path):
                with open(tokenizer_path, "rb") as f:
                    self.enc = pickle.load(f)
                self.eos_token_id = self.enc.encode_single_token("<|eos|>")
                self.pad_token_id = self.enc.encode_single_token("<|pad|>")
                self.source_token_id = self.enc.encode_single_token("<|source|>")
                self.target_token_id = self.enc.encode_single_token("<|target|>")
                self.bos_token_id = self.enc.encode_single_token("<|bos|>")

            def encode_text(self, text: str) -> list[int]:
                return self.enc.encode_ordinary(text)

            def decode_text(self, ids: list[int]) -> str:
                return self.enc.decode(ids)


        def _truncate(ids: list[int], max_len: int) -> list[int]:
            return ids[:max_len] if len(ids) > max_len else ids


        def _encode_prompt(tokenizer: Tokenizer, source: str) -> list[int]:
            source_ids = _truncate(tokenizer.encode_text(source), MAX_SEQ_LEN - 3)
            return [tokenizer.bos_token_id, tokenizer.source_token_id] + source_ids + [tokenizer.target_token_id]


        @dataclass
        class GPTConfig:
            sequence_len: int
            vocab_size: int
            n_layer: int
            n_head: int
            d_model: int
            mlp_ratio: float
            dropout: float


        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.float()
                norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return (x * norm).to(self.weight.dtype) * self.weight


        class CausalSelfAttention(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                assert config.d_model % config.n_head == 0
                self.n_head = config.n_head
                self.head_dim = config.d_model // config.n_head
                self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
                self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
                self.dropout = config.dropout

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, C = x.shape
                qkv = self.qkv(x)
                q, k, v = qkv.split(C, dim=-1)
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return self.proj(y)


        class MLP(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                hidden = int(config.d_model * config.mlp_ratio)
                self.fc = nn.Linear(config.d_model, hidden, bias=False)
                self.proj = nn.Linear(hidden, config.d_model, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(F.gelu(self.fc(x), approximate="tanh"))


        class Block(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                self.attn_norm = RMSNorm(config.d_model)
                self.attn = CausalSelfAttention(config)
                self.mlp_norm = RMSNorm(config.d_model)
                self.mlp = MLP(config)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.attn(self.attn_norm(x))
                x = x + self.mlp(self.mlp_norm(x))
                return x


        class GPT(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                self.config = config
                self.wte = nn.Embedding(config.vocab_size, config.d_model)
                self.wpe = nn.Embedding(config.sequence_len, config.d_model)
                self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
                self.norm = RMSNorm(config.d_model)
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
                self.lm_head.weight = self.wte.weight
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, idx: torch.Tensor) -> torch.Tensor:
                B, T = idx.shape
                if T > self.config.sequence_len:
                    raise ValueError(
                        f"Sequence length {{T}} exceeds configured limit {{self.config.sequence_len}}"
                    )
                pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
                x = self.wte(idx) + self.wpe(pos)[None, :, :]
                x = self.dropout(x)
                for block in self.blocks:
                    x = block(x)
                x = self.norm(x)
                return self.lm_head(x).float()


        def _decode_generated(tokenizer: Tokenizer, sequence: torch.Tensor, prompt_len: int) -> str:
            ids = sequence[prompt_len:].tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            ids = [token for token in ids if token != tokenizer.pad_token_id]
            return tokenizer.decode_text(ids).strip()


        @torch.no_grad()
        def generate_predictions(model: GPT, tokenizer: Tokenizer, rows: list[dict], batch_size: int = 64) -> list[str]:
            device = next(model.parameters()).device
            predictions = []
            autocast_enabled = device.type == "cuda"

            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                prompt_lens = [len(item["prompt_ids"]) for item in batch]
                max_prompt_len = max(prompt_lens)
                input_ids = torch.full(
                    (len(batch), max_prompt_len),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                for row_idx, item in enumerate(batch):
                    prompt_ids = torch.tensor(item["prompt_ids"], dtype=torch.long, device=device)
                    input_ids[row_idx, :prompt_ids.numel()] = prompt_ids

                lengths = torch.tensor(prompt_lens, dtype=torch.long, device=device)
                finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
                max_total_len = min(MAX_SEQ_LEN, max_prompt_len + MAX_TARGET_TOKENS)
                row_indices = torch.arange(len(batch), device=device)

                for _ in range(MAX_TARGET_TOKENS):
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ):
                        logits = model(input_ids)
                    next_token_logits = logits[row_indices, lengths - 1]
                    next_tokens = next_token_logits.argmax(dim=-1)
                    next_tokens = torch.where(
                        finished,
                        torch.full_like(next_tokens, tokenizer.pad_token_id),
                        next_tokens,
                    )

                    if input_ids.size(1) < max_total_len:
                        pad_col = torch.full(
                            (len(batch), 1),
                            tokenizer.pad_token_id,
                            dtype=torch.long,
                            device=device,
                        )
                        input_ids = torch.cat([input_ids, pad_col], dim=1)

                    can_write = (~finished) & (lengths < input_ids.size(1))
                    input_ids[row_indices[can_write], lengths[can_write]] = next_tokens[can_write]
                    lengths = lengths + can_write.long()
                    finished |= next_tokens.eq(tokenizer.eos_token_id)

                    if finished.all() or lengths.max().item() >= max_total_len:
                        break

                for row_idx, item in enumerate(batch):
                    sequence = input_ids[row_idx, : lengths[row_idx]]
                    predictions.append(_decode_generated(tokenizer, sequence, item["prompt_len"]))

            return predictions


        def load_test_rows(tokenizer: Tokenizer) -> tuple[list[dict], pd.DataFrame, str]:
            test_df = pd.read_csv(TEST_PATH)
            sample_df = pd.read_csv(SAMPLE_PATH)
            schema = _detect_schema(test_df, sample_df)
            id_col = schema["id_col"]

            rows = []
            if id_col and id_col in test_df.columns and id_col in sample_df.columns:
                if test_df[id_col].duplicated().any():
                    raise ValueError(f"test.csv contains duplicate ids in column '{{id_col}}'")
                test_by_id = test_df.set_index(id_col, drop=False)
                missing_ids = sample_df.loc[~sample_df[id_col].isin(test_by_id.index), id_col].tolist()
                if missing_ids:
                    preview = missing_ids[:5]
                    raise ValueError(
                        f"sample_submission.csv contains ids not present in test.csv: {{preview}}"
                    )

                for _, submission_row in sample_df.iterrows():
                    test_row = test_by_id.loc[submission_row[id_col]]
                    source = _normalize_text(test_row[schema["source_col"]])
                    prompt_ids = _encode_prompt(tokenizer, source)
                    rows.append(
                        {{
                            "id": _normalize_text(submission_row[id_col]),
                            "prompt_ids": prompt_ids,
                            "prompt_len": len(prompt_ids),
                        }}
                    )
            else:
                if len(test_df) != len(sample_df):
                    raise ValueError(
                        f"test.csv rows {{len(test_df)}} do not match sample rows {{len(sample_df)}}"
                    )
                for row_idx in range(len(sample_df)):
                    row = test_df.iloc[row_idx]
                    source = _normalize_text(row[schema["source_col"]])
                    prompt_ids = _encode_prompt(tokenizer, source)
                    rows.append(
                        {{
                            "id": str(row_idx),
                            "prompt_ids": prompt_ids,
                            "prompt_len": len(prompt_ids),
                        }}
                    )

            return rows, sample_df, schema["submission_col"]


        bundle_metrics = pd.read_json(METRICS_PATH, typ="series")
        print(bundle_metrics)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        config = GPTConfig(**checkpoint["config"])
        tokenizer = Tokenizer(TOKENIZER_PATH)
        model = GPT(config)
        model.load_state_dict(checkpoint["model_state"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print(f"Running inference on {{device}}")

        test_rows, submission, submission_col = load_test_rows(tokenizer)
        predictions = generate_predictions(model, tokenizer, test_rows)
        predictions = [_normalize_prediction_text(prediction) for prediction in predictions]

        if len(predictions) != len(submission):
            raise ValueError(
                f"Prediction count {{len(predictions)}} does not match sample rows {{len(submission)}}"
            )

        submission[submission_col] = predictions
        if submission[submission_col].isna().any():
            raise ValueError("Submission contains null prediction values.")
        if (submission[submission_col].astype(str).str.strip() == "").any():
            raise ValueError("Submission contains blank prediction values.")
        if list(submission.columns) != list(pd.read_csv(SAMPLE_PATH, nrows=0).columns):
            raise ValueError(
                f"Submission columns {{list(submission.columns)}} do not match sample columns "
                f"{{list(pd.read_csv(SAMPLE_PATH, nrows=0).columns)}}"
            )
        submission.to_csv(OUTPUT_PATH, index=False)
        print(f"Wrote {{OUTPUT_PATH}}")
        print(f"Submission rows: {{len(submission)}} | columns: {{list(submission.columns)}}")
        print(submission.head())
        """
    ).strip() + "\n"


def _build_notebook(dataset_slug: str, bundle_metrics: dict) -> dict:
    notebook_code = _build_notebook_code(dataset_slug, bundle_metrics)
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "overview",
                "metadata": {},
                "source": [
                    "# Deep Past submission notebook\n",
                    "\n",
                    "Loads the locally exported best autoresearch checkpoint, generates `submission.csv`, and leaves it in `/kaggle/working` for Kaggle submission.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "generate-submission",
                "metadata": {},
                "outputs": [],
                "source": notebook_code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Kaggle notebook assets for the current best model")
    parser.add_argument("--source-dir", type=Path, default=BEST_MODEL_DIR, help="Directory containing the best local checkpoint bundle")
    parser.add_argument("--output-dir", type=Path, default=KAGGLE_OUTPUT_DIR, help="Where to write the Kaggle dataset and notebook folders")
    parser.add_argument("--dataset-slug", help="Kaggle dataset slug in the form owner/dataset-name")
    parser.add_argument("--kernel-slug", help="Kaggle notebook slug in the form owner/notebook-name")
    parser.add_argument(
        "--enable-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the generated Kaggle notebook should request a GPU runtime",
    )
    args = parser.parse_args()

    dataset_slug = _resolve_owner_slug(args.dataset_slug, "autoresearch-deep-past-best")
    kernel_slug = _resolve_owner_slug(args.kernel_slug, "autoresearch-deep-past-submit")

    source_dir = args.source_dir
    checkpoint_path = source_dir / CHECKPOINT_NAME
    tokenizer_path = source_dir / TOKENIZER_NAME
    metrics_path = source_dir / METRICS_NAME
    missing = [path.name for path in [checkpoint_path, tokenizer_path, metrics_path] if not path.exists()]
    if missing:
        raise SystemExit(
            "Best-model bundle is incomplete. Missing: "
            + ", ".join(missing)
            + ". Run `uv run train.py` once to materialize `results/best_model/`."
        )

    bundle_metrics = _load_json(metrics_path)
    output_dir = args.output_dir
    dataset_dir = output_dir / "dataset"
    kernel_dir = output_dir / "kernel"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, dataset_dir / CHECKPOINT_NAME)
    shutil.copy2(tokenizer_path, dataset_dir / TOKENIZER_NAME)
    shutil.copy2(metrics_path, dataset_dir / METRICS_NAME)

    dataset_metadata = {
        "title": dataset_slug.split("/", 1)[-1],
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    _write_json(dataset_dir / "dataset-metadata.json", dataset_metadata)

    notebook = _build_notebook(dataset_slug, bundle_metrics)
    _write_json(kernel_dir / "submission_notebook.ipynb", notebook)

    kernel_metadata = {
        "id": kernel_slug,
        "title": kernel_slug.split("/", 1)[-1],
        "code_file": "submission_notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": args.enable_gpu,
        "enable_internet": False,
        "competition_sources": [COMPETITION_SLUG],
        "dataset_sources": [dataset_slug],
    }
    _write_json(kernel_dir / "kernel-metadata.json", kernel_metadata)

    readme = textwrap.dedent(
        f"""
        Current best checkpoint bundle:
          source: {source_dir}
          val_score: {bundle_metrics.get('val_score', 'n/a')}

        Kaggle dataset folder:
          {dataset_dir}

        Kaggle notebook folder:
          {kernel_dir}

        Kaggle dataset slug:
          {dataset_slug}

        Kaggle notebook slug:
          {kernel_slug}

        GPU enabled:
          {args.enable_gpu}

        Suggested flow:
          1. Create or update the dataset bundle:
             kaggle datasets version -p "{dataset_dir}" -m "Update best autoresearch checkpoint"
          2. Push the notebook:
             kaggle kernels push -p "{kernel_dir}"
          3. Open the notebook on Kaggle, click "Save Version" -> "Save & Run All".
          4. After the run finishes, open the notebook output and click "Submit" on submission.csv.

        If the Kaggle API CLI is not installed locally, upload `{dataset_dir}` as a dataset and
        import `{kernel_dir / 'submission_notebook.ipynb'}` manually in the Kaggle UI instead.
        """
    ).strip()
    (output_dir / "README.txt").write_text(readme + "\n")

    print(f"Prepared Kaggle dataset bundle at:  {dataset_dir}")
    print(f"Prepared Kaggle notebook bundle at: {kernel_dir}")
    print(f"Current best val_score:              {bundle_metrics.get('val_score', 'n/a')}")
    print()
    print(readme)


if __name__ == "__main__":
    main()
