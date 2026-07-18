# KV Cache Calculator

## 网页计算器

在线访问：[https://eco-sphere.github.io/kv-cache-calculator/](https://eco-sphere.github.io/kv-cache-calculator/)

直接用浏览器打开 `web/index.html`，或在仓库目录启动静态服务器：

```bash
python3 -m http.server 8000 --directory web
```

然后访问 `http://localhost:8000`。网页内置 9 个模型家族、52 个模型，并支持 TP、KV/Indexer 精度、Draft KV 和 Qwen Linear Attention runtime state 等模型相关选项。

### 网页支持范围

- DeepSeek：V4 Pro、V4 Flash、V3.2、V3、R1
- GLM：GLM-5、GLM-5.1、GLM-5.2
- Kimi：K2.5、K2.6
- Qwen：Qwen3.6、Qwen3.5、Qwen3、Qwen2.5 共 23 个模型
- Llama：Llama 3.1 8B/70B、Llama 3.3 70B
- Gemma：Gemma 4 E2B、E4B、26B-A4B、31B
- Cohere：Command R、R+、R7B、A、A Plus
- MiMo：MiMo-V2.5、MiMo-V2.5-Pro
- MiniMax：M2、M2.1、M2.5、M2.7、M3

网页覆盖 Standard MHA/GQA、MLA、DSA/MLA + Indexer、Qwen Linear/Full Hybrid、Full/Sliding GQA、MiniMax MSA 和 DeepSeek V4 Hybrid 七套容量公式。模型数据和公式来源见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

Python CLI 根据 HuggingFace 风格的 `config.json` **估算 decoder KV cache 占用字节数**，用于容量规划或和实测对照。CLI 支持以下两类结构，由配置自动分流。

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

- 静态网页全部 52 个模型和 TP 规则：`node --test tests/web_calculator.test.mjs`
- 用例从 **ModelScope** 拉取对应模型的 `config.json`，期望值集中在 **`tests/kv_cache_expected.json`**；上游 config 变更时需更新该 JSON。
- 无网或 CI 可跳过：`SKIP_MODELSCOPE=1 pytest tests/`
- 在 **`kv_cache_calculator`** 目录下：`python3 -m pytest tests/ -q`

依赖见 **`requirements.txt`** / **`requirements-dev.txt`**；跑非跳过用例时需网络以访问 ModelScope。

## 仓库布局

```
web/                     # 零依赖静态网页
  index.html             # 页面结构
  assets/
    styles.css           # 页面样式
    models.js            # 52 个静态模型配置
    calculator-core.js   # 七套容量计算公式
    script.js            # TP 分布、页面交互和渲染
kv_cache_size.py         # Python CLI 与计算 API
tests/
  web_calculator.test.mjs
  test_kv_cache_size.py
  kv_cache_expected.json
requirements.txt         # CLI 运行依赖
requirements-dev.txt     # 测试依赖
requirements-excel.txt   # 可选 Excel 依赖
pytest.ini               # pytest 配置
README.md                 # 项目说明
```

其他脚本（如 `build_kv_cache_excel.py`）若存在，与核心估算逻辑无强绑定，以各自脚本说明为准。
