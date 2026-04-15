#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from typing import Any


DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,  # indicative; depends on actual KV cache format
    "fp8": 1,
    "int8": 1,
}


def bytes_to_str(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.3f} {u}"
        x /= 1024.0
    return f"{x:.3f} TiB"


def _model_type_and_architectures(cfg: dict, text_cfg: dict) -> tuple[str | None, list]:
    mt = text_cfg.get("model_type") or cfg.get("model_type")
    arch = cfg.get("architectures") or text_cfg.get("architectures") or []
    return mt, arch if isinstance(arch, list) else []


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _resolve_dtype_bytes(text_cfg: dict, dtype_override: str | None) -> tuple[str, int]:
    dtype_resolved = (dtype_override or text_cfg.get("dtype", "float16")).lower()
    if dtype_resolved not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype={dtype_resolved}. Known: {sorted(DTYPE_BYTES.keys())}")
    return dtype_resolved, int(DTYPE_BYTES[dtype_resolved])


class BaseKVCacheSpec(ABC):
    """Shared: resolved dtype, token/batch counts, and report hook."""

    def __init__(
        self,
        *,
        num_tokens: int,
        batch: int,
        dtype_resolved: str,
        bytes_per_elem: int,
    ) -> None:
        self.num_tokens = int(num_tokens)
        self.batch = int(batch)
        self.dtype_resolved = dtype_resolved
        self.bytes_per_elem = bytes_per_elem

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Structured result (same keys as historical ``estimate_kv_cache_bytes``)."""

    @abstractmethod
    def print_report(
        self,
        *,
        model_source: str,
        num_tokens: int,
        batch: int,
        indexer_layout: str,
        cfg: dict,
        text_cfg: dict,
    ) -> None:
        """Human-readable estimate (stdout)."""


class GlmMLA(BaseKVCacheSpec):
    """GLM-5.1 / glm_moe_dsa: MLA latent KV + optional sparse indexer cache."""

    @staticmethod
    def supports(cfg: dict, text_cfg: dict) -> bool:
        model_type, arch = _model_type_and_architectures(cfg, text_cfg)
        if model_type == "glm_moe_dsa":
            return True
        return any(a == "GlmMoeDsaForCausalLM" for a in arch)

    @staticmethod
    def _mla_kv_bytes_per_token(
        *, batch: int, layers: int, kv_lora_rank: int, qk_rope_head_dim: int, bytes_per_elem: int
    ) -> int:
        return batch * layers * (kv_lora_rank + qk_rope_head_dim) * bytes_per_elem

    @staticmethod
    def _indexer_cache_bytes_per_token(
        *,
        batch: int,
        layers: int,
        index_head_dim: int,
        layout: str,
        quant_block_size: int = 128,
    ) -> int:
        if layout == "packed_uint8":
            head_size = index_head_dim + (index_head_dim // quant_block_size) * 4
            return batch * layers * head_size * 1
        if layout == "bf16":
            return batch * layers * index_head_dim * 2
        if layout == "sparse_c8":
            return batch * layers * (index_head_dim + 2)
        raise ValueError(
            f"Unknown indexer layout={layout!r}; use packed_uint8 | bf16 | sparse_c8"
        )

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        text_cfg: dict,
        *,
        num_tokens: int,
        batch: int,
        dtype_resolved: str,
        bytes_per_elem: int,
        indexer_layout: str,
    ) -> GlmMLA:
        layers = int(text_cfg["num_hidden_layers"])
        kv_lora_rank = int(text_cfg.get("kv_lora_rank", cfg.get("kv_lora_rank", 0)))
        qk_rope_head_dim = int(text_cfg.get("qk_rope_head_dim", cfg.get("qk_rope_head_dim", 0)))
        if kv_lora_rank <= 0 or qk_rope_head_dim <= 0:
            raise ValueError(
                "glm_moe_dsa config missing kv_lora_rank or qk_rope_head_dim; "
                "cannot estimate MLA KV size."
            )
        mla_per_token = cls._mla_kv_bytes_per_token(
            batch=batch,
            layers=layers,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            bytes_per_elem=bytes_per_elem,
        )
        index_topk = text_cfg.get("index_topk", cfg.get("index_topk"))
        indexer_per_token = 0
        index_head_dim = 0
        if index_topk is not None:
            index_head_dim = int(text_cfg.get("index_head_dim", cfg.get("index_head_dim", 0)))
            if index_head_dim > 0:
                indexer_per_token = cls._indexer_cache_bytes_per_token(
                    batch=batch,
                    layers=layers,
                    index_head_dim=index_head_dim,
                    layout=str(indexer_layout),
                )
        per_token = mla_per_token + indexer_per_token
        total = per_token * int(num_tokens)
        model_type, arch = _model_type_and_architectures(cfg, text_cfg)
        return cls(
            num_tokens=num_tokens,
            batch=batch,
            dtype_resolved=dtype_resolved,
            bytes_per_elem=bytes_per_elem,
            layers=layers,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            mla_per_token_bytes=mla_per_token,
            indexer_per_token_bytes=indexer_per_token,
            index_topk=index_topk,
            index_head_dim=index_head_dim,
            indexer_layout=str(indexer_layout),
            per_token_bytes=per_token,
            total_bytes=total,
            model_type=model_type,
            architectures=arch,
        )

    def __init__(
        self,
        *,
        num_tokens: int,
        batch: int,
        dtype_resolved: str,
        bytes_per_elem: int,
        layers: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        mla_per_token_bytes: int,
        indexer_per_token_bytes: int,
        index_topk: Any,
        index_head_dim: int,
        indexer_layout: str,
        per_token_bytes: int,
        total_bytes: int,
        model_type: str | None,
        architectures: list,
    ) -> None:
        super().__init__(
            num_tokens=num_tokens,
            batch=batch,
            dtype_resolved=dtype_resolved,
            bytes_per_elem=bytes_per_elem,
        )
        self.layers = layers
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.mla_per_token_bytes = mla_per_token_bytes
        self.indexer_per_token_bytes = indexer_per_token_bytes
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.indexer_layout = indexer_layout
        self.per_token_bytes = per_token_bytes
        self.total_bytes = total_bytes
        self.model_type = model_type
        self.architectures = architectures

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "glm_mla",
            "model_type": self.model_type,
            "architectures": self.architectures,
            "dtype": self.dtype_resolved,
            "bytes_per_elem": self.bytes_per_elem,
            "layers": self.layers,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "mla_per_token_bytes": self.mla_per_token_bytes,
            "indexer_per_token_bytes": self.indexer_per_token_bytes,
            "index_topk": self.index_topk,
            "index_head_dim": self.index_head_dim,
            "indexer_layout": self.indexer_layout,
            "per_token_bytes": self.per_token_bytes,
            "total_bytes": self.total_bytes,
        }

    def print_report(
        self,
        *,
        model_source: str,
        num_tokens: int,
        batch: int,
        indexer_layout: str,
        cfg: dict,
        text_cfg: dict,
    ) -> None:
        print("=== KV cache size (GLM-5.1 / glm_moe_dsa, MLA + optional indexer) ===")
        print(f"model: {model_source}")
        print(f"model_type: {self.model_type!r}  architectures: {self.architectures!r}")
        print(f"num_tokens (N): {num_tokens}")
        print(f"batch (B): {batch}")
        print(f"layers (L): {self.layers}")
        print(f"kv_lora_rank: {self.kv_lora_rank}  qk_rope_head_dim: {self.qk_rope_head_dim}")
        print(f"dtype (MLA latent): {self.dtype_resolved}  (bytes/elem={self.bytes_per_elem})")
        if self.indexer_per_token_bytes:
            print(
                f"index_topk: {self.index_topk}  index_head_dim: {self.index_head_dim}  "
                f"indexer_layout: {indexer_layout}"
            )
        n_mtp = text_cfg.get("num_nextn_predict_layers", cfg.get("num_nextn_predict_layers", 0))
        if int(n_mtp or 0) > 0:
            print(
                f"note: num_nextn_predict_layers={n_mtp} (MTP may add separate cache in some runtimes; not included)"
            )
        print()
        print(f"mla_per_token_bytes:      {self.mla_per_token_bytes}  ({bytes_to_str(self.mla_per_token_bytes)})")
        if self.indexer_per_token_bytes:
            print(
                f"indexer_per_token_bytes:  {self.indexer_per_token_bytes}  "
                f"({bytes_to_str(self.indexer_per_token_bytes)})"
            )
        print(f"per_token_bytes (sum):    {self.per_token_bytes}  ({bytes_to_str(self.per_token_bytes)})")
        print(f"total_bytes:              {self.total_bytes}  ({bytes_to_str(self.total_bytes)})")
        print()
        print("Formula:")
        print("  mla_per_token = B * L * (kv_lora_rank + qk_rope_head_dim) * bytes_per_elem")
        if self.indexer_per_token_bytes:
            if indexer_layout == "packed_uint8":
                print("  indexer_per_token = B * L * (D_idx + (D_idx//128)*4) * 1  # packed_uint8")
            elif indexer_layout == "bf16":
                print("  indexer_per_token = B * L * D_idx * 2  # bf16")
            else:
                print("  indexer_per_token = B * L * (D_idx * 1 + 2)  # sparse_c8 int8 K + fp16 scale")
        print("  per_token     = mla_per_token + indexer_per_token")
        print("  total         = per_token * N")


class NormalAttention(BaseKVCacheSpec):
    """Decoder self-attention K/V cache: same formula covers MHA, GQA, and MQA (via ``num_key_value_heads``)."""

    @staticmethod
    def _kv_bytes_per_token(
        *, batch: int, layers: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int
    ) -> int:
        return 2 * batch * layers * num_kv_heads * head_dim * bytes_per_elem

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        text_cfg: dict,
        *,
        num_tokens: int,
        batch: int,
        dtype_resolved: str,
        bytes_per_elem: int,
    ) -> NormalAttention:
        layers = int(text_cfg["num_hidden_layers"])
        num_kv_heads = int(text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"]))
        num_q_heads = int(text_cfg["num_attention_heads"])
        if "head_dim" in text_cfg:
            head_dim = int(text_cfg["head_dim"])
        else:
            head_dim = int(text_cfg["hidden_size"]) // num_q_heads
        per_token = cls._kv_bytes_per_token(
            batch=batch,
            layers=layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bytes_per_elem=bytes_per_elem,
        )
        total = per_token * int(num_tokens)
        return cls(
            num_tokens=num_tokens,
            batch=batch,
            dtype_resolved=dtype_resolved,
            bytes_per_elem=bytes_per_elem,
            layers=layers,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            per_token_bytes=per_token,
            total_bytes=total,
        )

    def __init__(
        self,
        *,
        num_tokens: int,
        batch: int,
        dtype_resolved: str,
        bytes_per_elem: int,
        layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        per_token_bytes: int,
        total_bytes: int,
    ) -> None:
        super().__init__(
            num_tokens=num_tokens,
            batch=batch,
            dtype_resolved=dtype_resolved,
            bytes_per_elem=bytes_per_elem,
        )
        self.layers = layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.per_token_bytes = per_token_bytes
        self.total_bytes = total_bytes

    def attention_variant(self) -> str:
        """MHA: Hkv==Hq; MQA: Hkv==1; else GQA."""
        if self.num_kv_heads >= self.num_q_heads:
            return "MHA"
        if self.num_kv_heads == 1:
            return "MQA"
        return "GQA"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "standard",
            "dtype": self.dtype_resolved,
            "bytes_per_elem": self.bytes_per_elem,
            "layers": self.layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "per_token_bytes": self.per_token_bytes,
            "total_bytes": self.total_bytes,
        }

    def print_report(
        self,
        *,
        model_source: str,
        num_tokens: int,
        batch: int,
        indexer_layout: str,
        cfg: dict,
        text_cfg: dict,
    ) -> None:
        _ = indexer_layout, cfg, text_cfg  # unused; kept for uniform call site
        variant = self.attention_variant()
        print("=== KV cache size (NormalAttention: MHA / GQA / MQA) ===")
        print(f"model: {model_source}")
        print(f"num_tokens (N): {num_tokens}")
        print(f"batch (B): {batch}")
        print(f"layers (L): {self.layers}")
        print(
            f"attention: {variant}  (num_attention_heads={self.num_q_heads}, "
            f"num_key_value_heads={self.num_kv_heads})"
        )
        print(f"head_dim (D): {self.head_dim}")
        print(f"dtype: {self.dtype_resolved}  (bytes/elem={self.bytes_per_elem})")
        print()
        print(f"per_token_bytes: {self.per_token_bytes}  ({bytes_to_str(self.per_token_bytes)})")
        print(f"total_bytes:     {self.total_bytes}  ({bytes_to_str(self.total_bytes)})")
        print()
        print("Formula:")
        print("  per_token = 2 * B * L * Hkv * D * bytes_per_elem")
        print("  total     = per_token * N")


def build_kv_cache_spec(
    cfg: dict,
    *,
    num_tokens: int,
    batch: int = 1,
    dtype: str | None = None,
    indexer_layout: str = "bf16",
) -> GlmMLA | NormalAttention:
    text_cfg = _text_cfg(cfg)
    dtype_resolved, bytes_per_elem = _resolve_dtype_bytes(text_cfg, dtype)
    if GlmMLA.supports(cfg, text_cfg):
        return GlmMLA.from_config(
            cfg,
            text_cfg,
            num_tokens=num_tokens,
            batch=batch,
            dtype_resolved=dtype_resolved,
            bytes_per_elem=bytes_per_elem,
            indexer_layout=indexer_layout,
        )
    return NormalAttention.from_config(
        cfg,
        text_cfg,
        num_tokens=num_tokens,
        batch=batch,
        dtype_resolved=dtype_resolved,
        bytes_per_elem=bytes_per_elem,
    )


def estimate_kv_cache_bytes(
    cfg: dict,
    *,
    num_tokens: int,
    batch: int = 1,
    dtype: str | None = None,
    indexer_layout: str = "bf16",
) -> dict[str, Any]:
    """Compute KV cache byte totals from a loaded HF-style ``config`` dict (same rules as CLI)."""
    return build_kv_cache_spec(
        cfg,
        num_tokens=num_tokens,
        batch=batch,
        dtype=dtype,
        indexer_layout=indexer_layout,
    ).as_dict()


def resolve_config_path(model: str) -> tuple[str, str]:
    """Return (absolute path to config.json, human-readable source label).

    If ``model`` is an existing local directory, use ``<dir>/config.json``.
    Otherwise treat ``model`` as a ModelScope model_id and download only ``config.json``.
    """
    raw = model.strip()
    local = os.path.abspath(os.path.expanduser(raw))
    if os.path.isfile(local):
        raise SystemExit(
            "--model must be a local directory containing config.json, not a path to a single file."
        )
    if os.path.isdir(local):
        path = os.path.join(local, "config.json")
        if not os.path.isfile(path):
            raise SystemExit(f"no config.json under model directory: {local}")
        return path, path
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError as e:
        raise SystemExit(
            "modelscope is required when --model is not a local directory. "
            "Install: pip install modelscope"
        ) from e
    downloaded = model_file_download(raw, "config.json")
    if not downloaded or not os.path.isfile(downloaded):
        raise SystemExit(
            f"ModelScope did not yield a valid config.json for model_id={raw!r}"
        )
    label = f"modelscope:{raw}/config.json -> {downloaded}"
    return downloaded, label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        metavar="DIR_OR_MODEL_ID",
        help="Local model directory (uses config.json inside), else ModelScope model_id to fetch config.json only.",
    )
    ap.add_argument("--num_tokens", "-N", type=int, required=True, help="cached tokens (context + generated)")
    ap.add_argument("--batch", "-B", type=int, default=1)
    ap.add_argument("--dtype", default=None, help="override config dtype, e.g. float16/bfloat16")
    ap.add_argument(
        "--indexer-layout",
        default="bf16",
        choices=("bf16", "packed_uint8", "sparse_c8"),
        help="glm_moe_dsa indexer KV 布局：bf16=Ascend 常见；packed_uint8=CUDA DeepseekV32IndexerCache；"
        "sparse_c8=Ascend sparse C8（int8+fp16 scale）",
    )
    args = ap.parse_args()

    config_path, config_source = resolve_config_path(args.model)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    text_cfg = _text_cfg(cfg)
    spec = build_kv_cache_spec(
        cfg,
        num_tokens=int(args.num_tokens),
        batch=int(args.batch),
        dtype=args.dtype,
        indexer_layout=str(args.indexer_layout),
    )
    spec.print_report(
        model_source=config_source,
        num_tokens=int(args.num_tokens),
        batch=int(args.batch),
        indexer_layout=str(args.indexer_layout),
        cfg=cfg,
        text_cfg=text_cfg,
    )


if __name__ == "__main__":
    main()
