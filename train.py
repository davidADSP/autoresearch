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
from sacrebleu.metrics import BLEU, CHRF

from prepare import (
    MAX_PROMPT_TOKENS,
    MAX_TARGET_TOKENS,
    TIME_BUDGET,
    TOKENIZER_DIR,
    Tokenizer,
    ensure_prepared_data,
    evaluate_translation,
    get_metadata,
    make_dataloader,
    postprocess_translation_text,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    source_len: int
    target_len: int
    vocab_size: int
    encoder_layers: int
    decoder_layers: int
    n_head: int
    d_model: int
    mlp_ratio: float
    dropout: float
    pad_token_id: int
    eos_token_id: int
    target_token_id: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        norm = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_fp32 * norm).to(self.weight.dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float, causal: bool):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            causal_mask = causal_mask[None, None, :, :]
            attn_mask = causal_mask if attn_mask is None else (attn_mask & causal_mask)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.proj(y)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, tgt_len, channels = x.shape
        src_len = memory.size(1)
        q = self.q_proj(x).view(batch_size, tgt_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch_size, src_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch_size, src_len, self.n_head, self.head_dim).transpose(1, 2)
        attn_mask = None
        if memory_padding_mask is not None:
            attn_mask = memory_padding_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, tgt_len, channels)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = SelfAttention(config.d_model, config.n_head, config.dropout, causal=False)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = MLP(config.d_model, config.mlp_ratio)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), source_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn_norm = RMSNorm(config.d_model)
        self.self_attn = SelfAttention(config.d_model, config.n_head, config.dropout, causal=True)
        self.cross_attn_norm = RMSNorm(config.d_model)
        self.cross_attn = CrossAttention(config.d_model, config.n_head, config.dropout)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = MLP(config.d_model, config.mlp_ratio)

    def forward(self, x: torch.Tensor, decoder_mask: torch.Tensor, memory: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_norm(x), decoder_mask)
        x = x + self.cross_attn(self.cross_attn_norm(x), memory, source_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MBRSelector:
    def __init__(
        self,
        w_chrf: float = 0.60,
        w_bleu: float = 0.20,
        w_jaccard: float = 0.20,
        w_length: float = 0.08,
        pool_cap: int = 16,
    ):
        self.chrf_metric = CHRF(word_order=2)
        self.bleu_metric = BLEU(effective_order=True)
        self.w_chrf = w_chrf
        self.w_bleu = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length = w_length
        self.pool_cap = pool_cap
        self.pair_weight = max(w_chrf + w_bleu + w_jaccard, 1e-8)

    @staticmethod
    def _dedup(candidates: list[str]) -> list[str]:
        seen = set()
        deduped = []
        for candidate in candidates:
            cleaned = candidate.strip()
            if cleaned and cleaned not in seen:
                deduped.append(cleaned)
                seen.add(cleaned)
        return deduped

    def _pairwise_score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        chrf = float(self.chrf_metric.sentence_score(a, [b]).score)
        try:
            bleu = float(self.bleu_metric.sentence_score(a, [b]).score)
        except Exception:
            bleu = 0.0
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if a_tokens or b_tokens:
            jaccard = 100.0 * len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
        else:
            jaccard = 100.0
        return (self.w_chrf * chrf + self.w_bleu * bleu + self.w_jaccard * jaccard) / self.pair_weight

    @staticmethod
    def _length_bonus(lengths: list[int], idx: int) -> float:
        if not lengths:
            return 100.0
        ordered = sorted(lengths)
        median = float(ordered[len(ordered) // 2])
        sigma = max(5.0, median * 0.4)
        z = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z * z)

    def pick(self, candidates: list[str]) -> str:
        candidates = self._dedup(candidates)
        if self.pool_cap:
            candidates = candidates[: self.pool_cap]
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        lengths = [len(candidate.split()) for candidate in candidates]
        scores = []
        for i, candidate in enumerate(candidates):
            pairwise = 0.0
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                pairwise += self._pairwise_score(candidate, other)
            pairwise /= max(len(candidates) - 1, 1)
            total = pairwise + self.w_length * self._length_bonus(lengths, i)
            scores.append(total)
        best_idx = max(range(len(candidates)), key=lambda idx: scores[idx])
        return candidates[best_idx]


class TranslationTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.source_pos = nn.Embedding(config.source_len, config.d_model)
        self.target_pos = nn.Embedding(config.target_len, config.d_model)
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_layers)])
        self.encoder_norm = RMSNorm(config.d_model)
        self.decoder_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.mbr = MBRSelector()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        batch_size, source_len = source_ids.shape
        if source_len > self.config.source_len:
            raise ValueError(f"Source length {source_len} exceeds configured limit {self.config.source_len}")
        positions = torch.arange(source_len, device=source_ids.device, dtype=torch.long)
        x = self.wte(source_ids) + self.source_pos(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.encoder:
            x = block(x, source_mask)
        return self.encoder_norm(x)

    def decode(self, decoder_input_ids: torch.Tensor, memory: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        batch_size, target_len = decoder_input_ids.shape
        if target_len > self.config.target_len:
            raise ValueError(f"Target length {target_len} exceeds configured limit {self.config.target_len}")
        positions = torch.arange(target_len, device=decoder_input_ids.device, dtype=torch.long)
        decoder_mask = decoder_input_ids.ne(self.config.pad_token_id)
        x = self.wte(decoder_input_ids) + self.target_pos(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.decoder:
            x = block(x, decoder_mask, memory, source_mask)
        x = self.decoder_norm(x)
        return self.lm_head(x).float()

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        memory = self.encode(source_ids, source_mask)
        logits = self.decode(decoder_input_ids, memory, source_mask)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            label_smoothing=LABEL_SMOOTHING,
        )
        return loss

    def _decode_token_ids(self, token_ids: list[int], tokenizer: Tokenizer) -> str:
        cleaned = []
        for token_id in token_ids:
            if token_id == self.config.eos_token_id:
                break
            if token_id in {self.config.pad_token_id, self.config.target_token_id}:
                continue
            cleaned.append(token_id)
        return tokenizer.decode_text(cleaned).strip()

    def _length_penalty(self, length: int) -> float:
        return ((5.0 + max(length, 1)) / 6.0) ** BEAM_LENGTH_PENALTY

    def _decode_logits(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(decoder_input_ids, memory, source_mask)

    def _beam_search(
        self,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        tokenizer: Tokenizer,
        max_new_tokens: int,
    ) -> list[str]:
        beams: list[tuple[list[int], float, bool]] = [([self.config.target_token_id], 0.0, False)]

        for _ in range(max_new_tokens):
            unfinished = [beam for beam in beams if not beam[2]]
            if not unfinished:
                break

            max_len = max(len(tokens) for tokens, _, _ in unfinished)
            decoder_input_ids = torch.full(
                (len(unfinished), max_len),
                self.config.pad_token_id,
                dtype=torch.long,
                device=memory.device,
            )
            for row_idx, (tokens, _, _) in enumerate(unfinished):
                decoder_input_ids[row_idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=memory.device)

            expanded_memory = memory.expand(len(unfinished), -1, -1)
            expanded_mask = source_mask.expand(len(unfinished), -1)
            logits = self._decode_logits(decoder_input_ids, expanded_memory, expanded_mask)
            last_indices = decoder_input_ids.ne(self.config.pad_token_id).sum(dim=1) - 1
            next_token_logits = logits[torch.arange(len(unfinished), device=memory.device), last_indices]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            expanded_beams: list[tuple[list[int], float, bool]] = []
            unfinished_idx = 0
            for tokens, score, finished in beams:
                if finished:
                    expanded_beams.append((tokens, score, True))
                    continue
                top_log_probs, top_ids = torch.topk(log_probs[unfinished_idx], k=NUM_BEAMS, dim=-1)
                for token_id, token_log_prob in zip(top_ids.tolist(), top_log_probs.tolist()):
                    new_tokens = tokens + [token_id]
                    expanded_beams.append((new_tokens, score + token_log_prob, token_id == self.config.eos_token_id))
                unfinished_idx += 1

            def beam_score(item: tuple[list[int], float, bool]) -> float:
                tokens, score, _ = item
                generated_len = max(len(tokens) - 1, 1)
                return score / self._length_penalty(generated_len)

            beams = sorted(expanded_beams, key=beam_score, reverse=True)[:NUM_BEAMS]

        beams = sorted(
            beams,
            key=lambda item: item[1] / self._length_penalty(max(len(item[0]) - 1, 1)),
            reverse=True,
        )[:NUM_BEAM_CANDS]
        return [self._decode_token_ids(tokens[1:], tokenizer) for tokens, _, _ in beams]

    def _sample_top_p(self, logits: torch.Tensor, temperature: float) -> int:
        logits = logits / max(temperature, 1e-4)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= SAMPLE_TOP_P
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sampled = torch.multinomial(filtered, num_samples=1).item()
        return int(sorted_idx[sampled].item())

    def _sample_sequence(
        self,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        tokenizer: Tokenizer,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        tokens = [self.config.target_token_id]
        for _ in range(max_new_tokens):
            decoder_input_ids = torch.tensor(tokens, dtype=torch.long, device=memory.device)[None, :]
            logits = self._decode_logits(decoder_input_ids, memory, source_mask)
            next_token_id = self._sample_top_p(logits[0, -1], temperature=temperature)
            tokens.append(next_token_id)
            if next_token_id == self.config.eos_token_id:
                break
        return self._decode_token_ids(tokens[1:], tokenizer)

    @torch.no_grad()
    def generate_translations(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        tokenizer: Tokenizer,
        max_new_tokens: int,
    ) -> list[str]:
        predictions = []
        autocast_enabled = source_ids.device.type == "cuda"
        was_training = self.training
        self.eval()

        for row_idx in range(source_ids.size(0)):
            single_source_ids = source_ids[row_idx : row_idx + 1]
            single_source_mask = source_mask[row_idx : row_idx + 1]
            with torch.amp.autocast(
                device_type=single_source_ids.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                memory = self.encode(single_source_ids, single_source_mask)
                candidates = self._beam_search(memory, single_source_mask, tokenizer, max_new_tokens=max_new_tokens)
                if USE_SAMPLING:
                    for temperature in SAMPLE_TEMPERATURES:
                        for _ in range(NUM_SAMPLE_PER_TEMP):
                            candidates.append(
                                self._sample_sequence(
                                    memory,
                                    single_source_mask,
                                    tokenizer,
                                    temperature=temperature,
                                    max_new_tokens=max_new_tokens,
                                )
                            )
            candidates = [postprocess_translation_text(candidate) for candidate in candidates]
            chosen = self.mbr.pick(candidates)
            predictions.append(chosen or "The tablet is too damaged to translate.")

        if was_training:
            self.train()
        return predictions


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
ENCODER_LAYERS = 3
DECODER_LAYERS = 3
D_MODEL = 320
N_HEAD = 8
MLP_RATIO = 4.0
DROPOUT = 0.15

# Optimization
BATCH_SIZE = 96
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.98
WARMUP_RATIO = 0.08
MIN_LR_FRAC = 0.15
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.05

# Validation decoding
NUM_BEAMS = 4
NUM_BEAM_CANDS = 3
BEAM_LENGTH_PENALTY = 1.0
USE_SAMPLING = True
SAMPLE_TEMPERATURES = (0.70, 1.00)
NUM_SAMPLE_PER_TEMP = 2
SAMPLE_TOP_P = 0.90

# Runtime
EVAL_BATCH_SIZE = 32
VAL_LOSS_BATCH_SIZE = 64
VAL_INTERVAL = 60.0
COMPILE_MODEL = False
WARMUP_STEPS = 4

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
    return LEARNING_RATE * (MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cosine)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_model_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in _unwrap_model(model).state_dict().items()}


@torch.no_grad()
def evaluate_val_loss(model: nn.Module, batch_size: int) -> dict[str, float]:
    device = next(model.parameters()).device
    autocast_enabled = device.type == "cuda"
    total_loss = 0.0
    total_tokens = 0

    for source_ids, source_mask, decoder_input_ids, targets, _ in make_dataloader(batch_size, "val", device=device.type):
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = model(source_ids, source_mask, decoder_input_ids, targets)
        target_tokens = (targets != -100).sum().item()
        total_loss += loss.item() * target_tokens
        total_tokens += target_tokens

    return {
        "loss": total_loss / max(total_tokens, 1),
        "num_tokens": total_tokens,
    }


def run_validation_snapshot(model: nn.Module) -> dict[str, float]:
    eval_model = _unwrap_model(model)
    was_training = model.training
    model.eval()
    val_loss_metrics = evaluate_val_loss(eval_model, batch_size=VAL_LOSS_BATCH_SIZE)
    if was_training:
        model.train()
    return {
        "val_loss": val_loss_metrics["loss"],
        "val_tokens": val_loss_metrics["num_tokens"],
    }


def _save_run_artifacts(
    model: nn.Module,
    config: ModelConfig,
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
        "model_family": "seq2seq_transformer",
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
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_target_tokens": MAX_TARGET_TOKENS,
            "vocab_size": tokenizer.get_vocab_size(),
            "decode": {
                "num_beams": NUM_BEAMS,
                "num_beam_candidates": NUM_BEAM_CANDS,
                "use_sampling": USE_SAMPLING,
                "sample_temperatures": list(SAMPLE_TEMPERATURES),
                "num_sample_per_temp": NUM_SAMPLE_PER_TEMP,
                "sample_top_p": SAMPLE_TOP_P,
            },
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
    for key in ("val_loss", "selected_step", "best_val_loss"):
        if key in metrics:
            run_metrics[key] = metrics[key]

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
    ensure_prepared_data()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_enabled = device.type == "cuda"

    tokenizer = Tokenizer.from_directory()
    metadata = get_metadata()
    vocab_size = tokenizer.get_vocab_size()

    config = ModelConfig(
        source_len=MAX_PROMPT_TOKENS,
        target_len=MAX_TARGET_TOKENS,
        vocab_size=vocab_size,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        n_head=N_HEAD,
        d_model=D_MODEL,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        target_token_id=tokenizer.target_token_id,
    )

    model = TranslationTransformer(config).to(device)
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
    source_ids, source_mask, decoder_input_ids, targets, epoch = next(train_loader)

    num_params = _unwrap_model(model).num_params()
    print(f"Dataset: train={metadata['train_examples']}, val={metadata['val_examples']}, test={metadata['test_examples']}")
    print(f"Model config: {asdict(config)}")
    print(f"Num params: {num_params / 1e6:.2f}M")
    print(f"Time budget: {TIME_BUDGET}s")

    t_start_training = time.time()
    total_training_time = 0.0
    smooth_loss = 0.0
    total_target_tokens = 0
    step = 0
    next_validation_at = VAL_INTERVAL
    best_val_loss = None
    best_val_snapshot = None
    best_model_state = None

    while True:
        torch.cuda.synchronize()
        step_start = time.time()

        for _ in range(GRAD_ACCUM_STEPS):
            progress = min(total_training_time / TIME_BUDGET, 1.0)
            lr = get_lr(progress)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = model(source_ids, source_mask, decoder_input_ids, targets)
            train_loss = loss.detach()
            (loss / GRAD_ACCUM_STEPS).backward()
            total_target_tokens += (targets != -100).sum().item()
            source_ids, source_mask, decoder_input_ids, targets, epoch = next(train_loader)

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
        tok_per_sec = int((BATCH_SIZE * MAX_TARGET_TOKENS * GRAD_ACCUM_STEPS) / max(dt, 1e-6))
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
        if step >= WARMUP_STEPS and total_training_time >= next_validation_at:
            print()
            validation = run_validation_snapshot(model)
            is_best_val = best_val_loss is None or validation["val_loss"] < best_val_loss
            if is_best_val:
                best_val_loss = validation["val_loss"]
                best_val_snapshot = {"step": step, **validation}
                best_model_state = _copy_model_state_to_cpu(model)
            best_suffix = " | best_val: yes" if is_best_val else ""
            print(
                f"val_check @ {100 * min(total_training_time / TIME_BUDGET, 1.0):5.1f}% | "
                f"val_loss: {validation['val_loss']:.4f} | "
                f"val_tokens: {validation['val_tokens']}"
                f"{best_suffix}"
            )
            next_validation_at += VAL_INTERVAL
        if step >= WARMUP_STEPS and total_training_time >= TIME_BUDGET:
            break

    print()

    selected_step = step
    if best_model_state is not None:
        _unwrap_model(model).load_state_dict(best_model_state)
        selected_step = best_val_snapshot["step"]
        print(
            f"restored_best_val_checkpoint: step {selected_step} | "
            f"best_val_loss: {best_val_snapshot['val_loss']:.4f}"
        )

    model.eval()
    val_loss_metrics = evaluate_val_loss(_unwrap_model(model), batch_size=VAL_LOSS_BATCH_SIZE)
    metrics = evaluate_translation(_unwrap_model(model), tokenizer, batch_size=EVAL_BATCH_SIZE, max_new_tokens=MAX_TARGET_TOKENS)
    metrics["val_loss"] = val_loss_metrics["loss"]
    metrics["selected_step"] = selected_step
    metrics["best_val_loss"] = best_val_snapshot["val_loss"] if best_val_snapshot is not None else val_loss_metrics["loss"]

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    startup_time = t_start_training - t_start

    print("---")
    print(f"selected_step:     {selected_step}")
    print(f"best_val_loss:     {metrics['best_val_loss']:.4f}")
    print(f"val_loss:         {metrics['val_loss']:.4f}")
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
