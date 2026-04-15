# KV Cache Calculator

根据 HuggingFace 风格的 `config.json` **估算 decoder KV cache 占用字节数**，用于容量规划或和实测对照。支持两类结构，由配置自动分流。

## 安装

```bash
pip install -r requirements.txt          # 仅跑 CLI（含 ModelScope 拉 config）
pip install -r requirements-dev.txt    # 再跑 pytest
# 可选：生成 Excel 表格
pip install -r requirements-excel.txt
```

## 支持的两种结构

| 类型 | 类名 | 适用模型 | 估算要点 |
|------|------|----------|----------|
| **GLM MLA + 可选 indexer** | `GlmMLA` | `model_type: glm_moe_dsa` 或 `GlmMoeDsaForCausalLM`（如 GLM-5.1） | MLA latent：`B·L·(kv_lora_rank + qk_rope_head_dim)·bytes`；若存在 `index_topk` 与 `index_head_dim`，再叠加 indexer（布局见 `--indexer-layout`） |
| **标准自注意力 K/V** | `NormalAttention` | 其余常见 LLM | **MHA / GQA / MQA** 共用同一式：`2·B·L·Hkv·D·bytes`（`Hkv = num_key_value_heads`，`D = head_dim`） |

说明：GlmMLA 路径下 **MTP 等额外 cache** 未计入；`fp8` 等仅为按字节宽度的粗算，与真实运行时量化布局可能不一致。

## 命令行

在仓库本目录下执行（需 Python 3.10+）：

```bash
# 本地权重目录（使用其中的 config.json）
python3 kv_cache_size.py --model /path/to/weights -N 8192

# 非本地目录则视为 ModelScope model_id，仅下载 config.json（需 pip install modelscope）
python3 kv_cache_size.py --model ZhipuAI/GLM-5.1 -N 8192

# 覆盖 config 中的 dtype（如 MLA 按 fp8 字节数估算）
python3 kv_cache_size.py --model ZhipuAI/GLM-5.1 -N 8192 --dtype fp8

# GLM indexer 布局：bf16 | packed_uint8 | sparse_c8
python3 kv_cache_size.py --model ZhipuAI/GLM-5.1 -N 8192 --indexer-layout bf16
```

常用参数：`--model`（必填）、`-N` / `--num_tokens`、`--batch`、`--dtype`、`--indexer-layout`。

## Python API

```python
from kv_cache_size import build_kv_cache_spec, estimate_kv_cache_bytes

# 字典结果（与 CLI 同一套公式）
out = estimate_kv_cache_bytes(cfg, num_tokens=8192, batch=1, dtype="fp8", indexer_layout="bf16")

# 或拿到 spec 再打印 / 扩展
spec = build_kv_cache_spec(cfg, num_tokens=8192, batch=1, dtype=None, indexer_layout="bf16")
d = spec.as_dict()
spec.print_report(model_source="...", num_tokens=8192, batch=1, indexer_layout="bf16", cfg=cfg, text_cfg=cfg.get("text_config", cfg))
```

## 测试

- 用例从 **ModelScope** 拉取对应模型的 `config.json`，期望值集中在 **`tests/kv_cache_expected.json`**；上游 config 变更时需更新该 JSON。
- 无网或 CI 可跳过：`SKIP_MODELSCOPE=1 pytest tests/`
- 在 **`kv_cache_calculator`** 目录下：`python3 -m pytest tests/ -q`

依赖见 **`requirements.txt`** / **`requirements-dev.txt`**；跑非跳过用例时需网络以访问 ModelScope。

## 仓库布局

```
kv_cache_size.py         # CLI + GlmMLA / NormalAttention + estimate_kv_cache_bytes 等
requirements.txt         # 运行 CLI（ModelScope）
requirements-dev.txt     # + pytest
requirements-excel.txt   # 可选：build_kv_cache_excel.py
tests/
  test_kv_cache_size.py
  kv_cache_expected.json
pytest.ini
README.md
```

其他脚本（如 `build_kv_cache_excel.py`）若存在，与核心估算逻辑无强绑定，以各自脚本说明为准。
