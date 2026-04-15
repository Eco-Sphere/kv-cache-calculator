#!/usr/bin/env python3
import argparse
import json
import os


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


def kv_cache_bytes_per_token(*, batch: int, layers: int, num_kv_heads: int, head_dim: int, bytes_per_elem: int) -> int:
    # K and V
    return 2 * batch * layers * num_kv_heads * head_dim * bytes_per_elem


def _model_type_and_architectures(cfg: dict, text_cfg: dict) -> tuple[str | None, list]:
    mt = text_cfg.get("model_type") or cfg.get("model_type")
    arch = cfg.get("architectures") or text_cfg.get("architectures") or []
    return mt, arch if isinstance(arch, list) else []


def is_glm_moe_dsa(cfg: dict, text_cfg: dict) -> bool:
    """ZhipuAI/GLM-5.1, zai-org/GLM-5.1, etc. — DeepSeek-style MLA + optional sparse indexer."""
    model_type, arch = _model_type_and_architectures(cfg, text_cfg)
    if model_type == "glm_moe_dsa":
        return True
    return any(a == "GlmMoeDsaForCausalLM" for a in arch)


def glm_mla_kv_bytes_per_token(
    *, batch: int, layers: int, kv_lora_rank: int, qk_rope_head_dim: int, bytes_per_elem: int
) -> int:
    # vLLM MLAAttentionSpec: one logical "head", head_size = kv_lora_rank + qk_rope_head_dim (no separate K/V doubling).
    return batch * layers * (kv_lora_rank + qk_rope_head_dim) * bytes_per_elem


def glm_indexer_cache_bytes_per_token(
    *,
    batch: int,
    layers: int,
    index_head_dim: int,
    layout: str = "bf16",
    quant_block_size: int = 128,
) -> int:
    """Indexer K cache bytes for one full forward step (all layers, batch B).

    layout:
      - bf16: Ascend / vLLM-Ascend 上常见：indexer K 与 MLA 同精度存（D_idx * 2 B/token/层）。
        285952 tokens × (89856 + 78×256) B ≈ 29.25 GiB，与实测一致。
      - packed_uint8: 上游 CUDA vLLM DeepseekV32IndexerCache（uint8 + 每块 fp32 scale）。
      - sparse_c8: vLLM-Ascend AscendMLAAttentionSpec.cache_sparse_c8（int8 K + fp16 scale/token）。
    """
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


def estimate_kv_cache_bytes(
    cfg: dict,
    *,
    num_tokens: int,
    batch: int = 1,
    dtype: str | None = None,
    indexer_layout: str = "bf16",
) -> dict:
    """Compute KV cache byte totals from a loaded HF-style ``config`` dict (same rules as CLI)."""
    text_cfg = cfg.get("text_config", cfg)
    layers = int(text_cfg["num_hidden_layers"])
    num_kv_heads = int(text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"]))

    if "head_dim" in text_cfg:
        head_dim = int(text_cfg["head_dim"])
    else:
        head_dim = int(text_cfg["hidden_size"]) // int(text_cfg["num_attention_heads"])

    dtype_resolved = (dtype or text_cfg.get("dtype", "float16")).lower()
    if dtype_resolved not in DTYPE_BYTES:
        raise ValueError(
            f"Unknown dtype={dtype_resolved}. Known: {sorted(DTYPE_BYTES.keys())}"
        )
    bytes_per_elem = int(DTYPE_BYTES[dtype_resolved])

    if is_glm_moe_dsa(cfg, text_cfg):
        kv_lora_rank = int(text_cfg.get("kv_lora_rank", cfg.get("kv_lora_rank", 0)))
        qk_rope_head_dim = int(text_cfg.get("qk_rope_head_dim", cfg.get("qk_rope_head_dim", 0)))
        if kv_lora_rank <= 0 or qk_rope_head_dim <= 0:
            raise ValueError(
                "glm_moe_dsa config missing kv_lora_rank or qk_rope_head_dim; "
                "cannot estimate MLA KV size."
            )
        mla_per_token = glm_mla_kv_bytes_per_token(
            batch=int(batch),
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
                indexer_per_token = glm_indexer_cache_bytes_per_token(
                    batch=int(batch),
                    layers=layers,
                    index_head_dim=index_head_dim,
                    layout=str(indexer_layout),
                )
        per_token = mla_per_token + indexer_per_token
        total = per_token * int(num_tokens)
        model_type, arch = _model_type_and_architectures(cfg, text_cfg)
        return {
            "mode": "glm_mla",
            "model_type": model_type,
            "architectures": arch,
            "dtype": dtype_resolved,
            "bytes_per_elem": bytes_per_elem,
            "layers": layers,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "mla_per_token_bytes": mla_per_token,
            "indexer_per_token_bytes": indexer_per_token,
            "index_topk": index_topk,
            "index_head_dim": index_head_dim,
            "indexer_layout": indexer_layout,
            "per_token_bytes": per_token,
            "total_bytes": total,
        }

    per_token = kv_cache_bytes_per_token(
        batch=int(batch),
        layers=layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_elem=bytes_per_elem,
    )
    total = per_token * int(num_tokens)
    return {
        "mode": "standard",
        "dtype": dtype_resolved,
        "bytes_per_elem": bytes_per_elem,
        "layers": layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "per_token_bytes": per_token,
        "total_bytes": total,
    }


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

    out = estimate_kv_cache_bytes(
        cfg,
        num_tokens=int(args.num_tokens),
        batch=int(args.batch),
        dtype=args.dtype,
        indexer_layout=str(args.indexer_layout),
    )
    text_cfg = cfg.get("text_config", cfg)

    if out["mode"] == "glm_mla":
        print("=== KV cache size (GLM-5.1 / glm_moe_dsa, MLA + optional indexer) ===")
        print(f"model: {config_source}")
        print(f"model_type: {out['model_type']!r}  architectures: {out['architectures']!r}")
        print(f"num_tokens (N): {args.num_tokens}")
        print(f"batch (B): {args.batch}")
        print(f"layers (L): {out['layers']}")
        print(f"kv_lora_rank: {out['kv_lora_rank']}  qk_rope_head_dim: {out['qk_rope_head_dim']}")
        print(f"dtype (MLA latent): {out['dtype']}  (bytes/elem={out['bytes_per_elem']})")
        if out["indexer_per_token_bytes"]:
            print(
                f"index_topk: {out['index_topk']}  index_head_dim: {out['index_head_dim']}  "
                f"indexer_layout: {args.indexer_layout}"
            )
        n_mtp = text_cfg.get("num_nextn_predict_layers", cfg.get("num_nextn_predict_layers", 0))
        if int(n_mtp or 0) > 0:
            print(
                f"note: num_nextn_predict_layers={n_mtp} (MTP may add separate cache in some runtimes; not included)"
            )
        print()
        mla_per_token = out["mla_per_token_bytes"]
        indexer_per_token = out["indexer_per_token_bytes"]
        per_token = out["per_token_bytes"]
        total = out["total_bytes"]
        print(f"mla_per_token_bytes:      {mla_per_token}  ({bytes_to_str(mla_per_token)})")
        if indexer_per_token:
            print(f"indexer_per_token_bytes:  {indexer_per_token}  ({bytes_to_str(indexer_per_token)})")
        print(f"per_token_bytes (sum):    {per_token}  ({bytes_to_str(per_token)})")
        print(f"total_bytes:              {total}  ({bytes_to_str(total)})")
        print()
        print("Formula:")
        print("  mla_per_token = B * L * (kv_lora_rank + qk_rope_head_dim) * bytes_per_elem")
        if indexer_per_token:
            if args.indexer_layout == "packed_uint8":
                print("  indexer_per_token = B * L * (D_idx + (D_idx//128)*4) * 1  # packed_uint8")
            elif args.indexer_layout == "bf16":
                print("  indexer_per_token = B * L * D_idx * 2  # bf16")
            else:
                print("  indexer_per_token = B * L * (D_idx * 1 + 2)  # sparse_c8 int8 K + fp16 scale")
        print("  per_token     = mla_per_token + indexer_per_token")
        print("  total         = per_token * N")
        return

    per_token = out["per_token_bytes"]
    total = out["total_bytes"]
    print("=== KV cache size (decoder self-attn) ===")
    print(f"model: {config_source}")
    print(f"num_tokens (N): {args.num_tokens}")
    print(f"batch (B): {args.batch}")
    print(f"layers (L): {out['layers']}")
    print(f"num_kv_heads (Hkv): {out['num_kv_heads']}")
    print(f"head_dim (D): {out['head_dim']}")
    print(f"dtype: {out['dtype']}  (bytes/elem={out['bytes_per_elem']})")
    print()
    print(f"per_token_bytes: {per_token}  ({bytes_to_str(per_token)})")
    print(f"total_bytes:     {total}  ({bytes_to_str(total)})")
    print()
    print("Formula:")
    print("  per_token = 2 * B * L * Hkv * D * bytes_per_elem")
    print("  total     = per_token * N")


if __name__ == "__main__":
    main()

