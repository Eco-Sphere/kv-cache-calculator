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

## 测试

- 静态网页全部 52 个模型和 TP 规则：`node --test tests/web_calculator.test.mjs`

## 反馈

如果你发现模型参数、计算假设或结果存在问题，欢迎提交 [Issue](https://github.com/Eco-Sphere/kv-cache-calculator/issues/new)。请附上模型与输入配置、预期结果、实际结果和可验证的数据来源。

## 仓库布局

```
web/                     # 零依赖静态网页
  index.html             # 页面结构
  assets/
    styles.css           # 页面样式
    models.js            # 52 个静态模型配置
    calculator-core.js   # 七套容量计算公式
    script.js            # TP 分布、页面交互和渲染
tests/
  web_calculator.test.mjs
README.md                 # 项目说明
```
