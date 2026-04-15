"""KV cache tests: config from ModelScope, expected values in ``kv_cache_expected.json``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kv_cache_size import estimate_kv_cache_bytes, resolve_config_path

_TESTS_DIR = Path(__file__).resolve().parent
_EXPECTED_PATH = _TESTS_DIR / "kv_cache_expected.json"


def _load_expected_cases():
    with open(_EXPECTED_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    return doc["cases"]


def _config_from_modelscope(model_id: str) -> dict:
    from modelscope.hub.file_download import model_file_download

    path = model_file_download(model_id, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


pytestmark = [
    pytest.mark.modelscope,
    pytest.mark.skipif(
        os.environ.get("SKIP_MODELSCOPE", "").lower() in ("1", "true", "yes"),
        reason="SKIP_MODELSCOPE set — skip tests that download config from ModelScope",
    ),
]


@pytest.mark.parametrize("case", _load_expected_cases(), ids=lambda c: c["id"])
def test_kv_cache_matches_expected_json(case: dict):
    """Hub ``config.json`` (ModelScope) + ``kv_cache_expected.json`` 中的 expect 字段逐项一致。"""
    cfg = _config_from_modelscope(case["model_id"])
    dtype = case.get("dtype")
    out = estimate_kv_cache_bytes(
        cfg,
        num_tokens=int(case["num_tokens"]),
        batch=int(case.get("batch", 1)),
        dtype=dtype,
        indexer_layout=str(case.get("indexer_layout", "bf16")),
    )
    expect = case["expect"]
    mismatches = []
    for key, want in expect.items():
        got = out.get(key)
        if got != want:
            mismatches.append(f"  {key!r}: got {got!r}, want {want!r}")
    if mismatches:
        pytest.fail(
            f"case {case['id']!r} (model_id={case['model_id']!r}):\n" + "\n".join(mismatches)
        )


def test_resolve_config_path_modelscope_id():
    """``resolve_config_path`` 对非本地目录应走 ModelScope 并返回磁盘上的 config.json。"""
    path, label = resolve_config_path("ZhipuAI/GLM-5.1")
    assert path.endswith("config.json")
    assert "modelscope:" in label and "ZhipuAI/GLM-5.1" in label
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("model_type") == "glm_moe_dsa"
