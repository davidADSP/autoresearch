"""
Autoresearch training script for Deep Past translation.

Usage:
    uv run train.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    MAX_SEQ_LEN,
    MAX_TARGET_TOKENS,
    TIME_BUDGET,
    TOKENIZER_DIR,
    Tokenizer,
    evaluate_translation,
    get_metadata,
    make_dataloader,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


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

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.config.sequence_len:
            raise ValueError(f"Sequence length {T} exceeds configured limit {self.config.sequence_len}")
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x).float()
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return loss


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
N_LAYER = 6
D_MODEL = 512
N_HEAD = 8
MLP_RATIO = 4.0
DROPOUT = 0.0

# Optimization
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.95
WARMUP_RATIO = 0.05
MIN_LR_FRAC = 0.10
MAX_GRAD_NORM = 1.0

# Runtime
EVAL_BATCH_SIZE = 64
COMPILE_MODEL = False
WARMUP_STEPS = 5

# Artifacts
RESULTS_DIR = Path("results")
LAST_RUN_DIR = RESULTS_DIR / "last_run"
BEST_MODEL_DIR = RESULTS_DIR / "best_model"
CHECKPOINT_NAME = "checkpoint.pt"
TOKENIZER_NAME = "tokenizer.pkl"
METRICS_NAME = "metrics.json"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def get_lr(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return LEARNING_RATE * progress / max(WARMUP_RATIO, 1e-8)
    decay_progress = (progress - WARMUP_RATIO) / max(1.0 - WARMUP_RATIO, 1e-8)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(decay_progress, 0.0), 1.0)))
    floor = MIN_LR_FRAC
    return LEARNING_RATE * (floor + (1.0 - floor) * cosine)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save_run_artifacts(
    model: nn.Module,
    config: GPTConfig,
    tokenizer: Tokenizer,
    dataset_metadata: dict,
    metrics: dict,
    total_training_time: float,
    startup_time: float,
    total_runtime: float,
    peak_vram_mb: float,
    total_target_tokens: int,
    step: int,
    num_params: int,
) -> tuple[bool, dict[str, Path]]:
    checkpoint = {
        "model_state": _unwrap_model(model).state_dict(),
        "config": asdict(config),
        "metrics": metrics,
        "dataset_metadata": dataset_metadata,
        "runtime": {
            "training_seconds": total_training_time,
            "startup_seconds": startup_time,
            "total_seconds": total_runtime,
            "peak_vram_mb": peak_vram_mb,
            "target_tokens_M": total_target_tokens / 1e6,
            "num_steps": step,
            "num_params_M": num_params / 1e6,
        },
        "inference": {
            "max_seq_len": MAX_SEQ_LEN,
            "max_target_tokens": MAX_TARGET_TOKENS,
            "vocab_size": tokenizer.get_vocab_size(),
        },
    }

    run_metrics = {
        "val_score": metrics["score"],
        "val_bleu": metrics["bleu"],
        "val_chrf": metrics["chrf"],
        "val_examples": metrics["num_examples"],
        "training_seconds": total_training_time,
        "startup_seconds": startup_time,
        "total_seconds": total_runtime,
        "peak_vram_mb": peak_vram_mb,
        "target_tokens_M": total_target_tokens / 1e6,
        "num_steps": step,
        "num_params_M": num_params / 1e6,
    }

    LAST_RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, LAST_RUN_DIR / CHECKPOINT_NAME)
    shutil.copy2(TOKENIZER_DIR / TOKENIZER_NAME, LAST_RUN_DIR / TOKENIZER_NAME)
    _write_json(LAST_RUN_DIR / METRICS_NAME, run_metrics)

    best_score = None
    best_metrics_path = BEST_MODEL_DIR / METRICS_NAME
    if best_metrics_path.exists():
        best_score = json.loads(best_metrics_path.read_text()).get("val_score")

    is_new_best = best_score is None or metrics["score"] > float(best_score)
    if is_new_best:
        BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LAST_RUN_DIR / CHECKPOINT_NAME, BEST_MODEL_DIR / CHECKPOINT_NAME)
        shutil.copy2(LAST_RUN_DIR / TOKENIZER_NAME, BEST_MODEL_DIR / TOKENIZER_NAME)
        shutil.copy2(LAST_RUN_DIR / METRICS_NAME, BEST_MODEL_DIR / METRICS_NAME)

    return is_new_best, {
        "last_run_dir": LAST_RUN_DIR,
        "best_model_dir": BEST_MODEL_DIR,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires a CUDA GPU.")

    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_enabled = device.type == "cuda"

    tokenizer = Tokenizer.from_directory()
    metadata = get_metadata()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        d_model=D_MODEL,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    )

    model = GPT(config).to(device)
    model.init_weights()
    if COMPILE_MODEL:
        model = torch.compile(model, dynamic=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )

    train_loader = make_dataloader(BATCH_SIZE, "train", device=device.type)
    x, y, epoch = next(train_loader)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: train={metadata['train_examples']}, val={metadata['val_examples']}, test={metadata['test_examples']}")
    print(f"Model config: {asdict(config)}")
    print(f"Num params: {num_params / 1e6:.2f}M")
    print(f"Time budget: {TIME_BUDGET}s")

    t_start_training = time.time()
    total_training_time = 0.0
    smooth_loss = 0.0
    total_target_tokens = 0
    step = 0

    while True:
        torch.cuda.synchronize()
        step_start = time.time()

        for micro_step in range(GRAD_ACCUM_STEPS):
            progress = min(total_training_time / TIME_BUDGET, 1.0)
            lr = get_lr(progress)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = model(x, y)
            train_loss = loss.detach()
            (loss / GRAD_ACCUM_STEPS).backward()
            total_target_tokens += (y != -100).sum().item()
            x, y, epoch = next(train_loader)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_end = time.time()
        dt = step_end - step_start
        if step >= WARMUP_STEPS:
            total_training_time += dt

        loss_value = train_loss.item()
        if math.isnan(loss_value) or math.isinf(loss_value) or loss_value > 100:
            print("FAIL")
            raise SystemExit(1)

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_value
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        pct_done = 100 * progress
        tok_per_sec = int((BATCH_SIZE * MAX_SEQ_LEN * GRAD_ACCUM_STEPS) / max(dt, 1e-6))
        remaining = max(0.0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:5.1f}%) | "
            f"loss: {debiased_loss:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
            f"grad: {float(grad_norm):.2f} | "
            f"tok/sec: {tok_per_sec:,} | "
            f"epoch: {epoch} | "
            f"remaining: {remaining:.0f}s   ",
            end="",
            flush=True,
        )

        step += 1
        if step >= WARMUP_STEPS and total_training_time >= TIME_BUDGET:
            break

    print()

    model.eval()
    metrics = evaluate_translation(model, tokenizer, batch_size=EVAL_BATCH_SIZE, max_new_tokens=MAX_TARGET_TOKENS)

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    startup_time = t_start_training - t_start

    print("---")
    print(f"val_score:        {metrics['score']:.6f}")
    print(f"val_bleu:         {metrics['bleu']:.6f}")
    print(f"val_chrf:         {metrics['chrf']:.6f}")
    print(f"val_examples:     {metrics['num_examples']}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"startup_seconds:  {startup_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"target_tokens_M:  {total_target_tokens / 1e6:.2f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")

    is_new_best, artifact_paths = _save_run_artifacts(
        model=model,
        config=config,
        tokenizer=tokenizer,
        dataset_metadata=metadata,
        metrics=metrics,
        total_training_time=total_training_time,
        startup_time=startup_time,
        total_runtime=t_end - t_start,
        peak_vram_mb=peak_vram_mb,
        total_target_tokens=total_target_tokens,
        step=step,
        num_params=num_params,
    )
    print(f"last_run_dir:     {artifact_paths['last_run_dir']}")
    best_suffix = " (updated)" if is_new_best else ""
    print(f"best_model_dir:   {artifact_paths['best_model_dir']}{best_suffix}")


if __name__ == "__main__":
    main()
