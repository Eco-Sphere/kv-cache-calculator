#!/usr/bin/env python3
import json
import argparse


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/weights/Qwen3-VL-8B-Instruct/config.json")
    ap.add_argument("--num_tokens", "-N", type=int, required=True, help="cached tokens (context + generated)")
    ap.add_argument("--batch", "-B", type=int, default=1)
    ap.add_argument("--dtype", default=None, help="override config dtype, e.g. float16/bfloat16")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    text_cfg = cfg.get("text_config", cfg)
    layers = int(text_cfg["num_hidden_layers"])
    num_kv_heads = int(text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"]))

    if "head_dim" in text_cfg:
        head_dim = int(text_cfg["head_dim"])
    else:
        head_dim = int(text_cfg["hidden_size"]) // int(text_cfg["num_attention_heads"])

    dtype = (args.dtype or text_cfg.get("dtype", "float16")).lower()
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype={dtype}. Known: {sorted(DTYPE_BYTES.keys())}")
    bytes_per_elem = int(DTYPE_BYTES[dtype])

    per_token = kv_cache_bytes_per_token(
        batch=int(args.batch),
        layers=layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_elem=bytes_per_elem,
    )
    total = per_token * int(args.num_tokens)

    print("=== KV cache size (decoder self-attn) ===")
    print(f"config: {args.config}")
    print(f"num_tokens (N): {args.num_tokens}")
    print(f"batch (B): {args.batch}")
    print(f"layers (L): {layers}")
    print(f"num_kv_heads (Hkv): {num_kv_heads}")
    print(f"head_dim (D): {head_dim}")
    print(f"dtype: {dtype}  (bytes/elem={bytes_per_elem})")
    print()
    print(f"per_token_bytes: {per_token}  ({bytes_to_str(per_token)})")
    print(f"total_bytes:     {total}  ({bytes_to_str(total)})")
    print()
    print("Formula:")
    print("  per_token = 2 * B * L * Hkv * D * bytes_per_elem")
    print("  total     = per_token * N")


if __name__ == "__main__":
    main()

