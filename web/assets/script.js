const MODELS = [
  {
    family: "MiniMax", id: "minimax-m3", label: "MiniMax M3", type: "minimax_msa",
    layers: 60, fullLayers: 3, sparseLayers: 57, heads: 4, dim: 128,
    indexDim: 128, indexQHeads: 4, indexKHeads: 1, indexBlockSize: 128,
    indexTopkBlocks: 16, indexLocalBlocks: 1, mtpModules: 7, nextNLayers: 1,
    max: 1048576, source: "MiniMaxAI/MiniMax-M3"
  },
  {
    family: "Qwen3.6", id: "qwen3.6-27b", label: "Qwen3.6-27B", type: "qwen_linear_full_hybrid",
    layers: 64, fullLayers: 16, linearLayers: 48, heads: 4, dim: 256,
    linearKeyHeads: 16, linearKeyDim: 128, linearValueHeads: 48,
    linearValueDim: 128, linearConvKernel: 4, mtpLayers: 1,
    max: 262144, source: "Qwen/Qwen3.6-27B"
  }
];

const $ = (id) => document.getElementById(id);
const families = [...new Set(MODELS.map((model) => model.family))];
families.forEach((family) => $("family").add(new Option(family, family)));

function modelsForFamily() {
  return MODELS.filter((model) => model.family === $("family").value);
}

function fillModels() {
  const previous = $("model").value;
  $("model").replaceChildren();
  modelsForFamily().forEach((model) => $("model").add(new Option(model.label, model.id)));
  if (modelsForFamily().some((model) => model.id === previous)) $("model").value = previous;
  syncModel();
}

function selected() {
  return MODELS.find((model) => model.id === $("model").value) || modelsForFamily()[0];
}

function formatBytes(bytes) {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 1 : value >= 10 ? 2 : 3)} ${units[unit]}`;
}

function syncTpOptions(model) {
  const select = $("tensor-parallel");
  const previous = Number(select.value);
  const options = Array.from({ length: model.heads }, (_, index) => index + 1)
    .filter((tp) => model.heads % tp === 0);
  select.replaceChildren(...options.map((tp) => new Option(String(tp), String(tp))));
  select.value = options.includes(previous) ? String(previous) : String(options[0]);
}

function syncModel() {
  const model = selected();
  $("tokens").value = Math.min(1024, model.max);
  syncTpOptions(model);
  $("indexer-control").hidden = model.type !== "minimax_msa";
  $("indexer").disabled = false;
  $("linear-control").hidden = model.type !== "qwen_linear_full_hybrid";
  $("linear").checked = model.type === "qwen_linear_full_hybrid";
  $("draft-control").hidden = true;
  $("draft").checked = false;
  calculate();
}

function renderFormula(rows) {
  $("formula-list").replaceChildren(...rows.map(([name, expression]) => {
    const row = document.createElement("div");
    row.className = "formula-row";
    row.innerHTML = `<span class="formula-name">${name}</span><span class="formula-equals">=</span><span class="formula-expression">${expression}</span>`;
    return row;
  }));
}

function renderBreakdown(rows) {
  $("breakdown").replaceChildren(...rows.map(([name, value, help]) => {
    const row = document.createElement("div");
    row.className = "breakdown-row";
    const label = document.createElement("span");
    label.textContent = name;
    if (help) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "kv-help";
      button.textContent = "?";
      button.setAttribute("aria-label", help);
      button.dataset.tooltip = help;
      label.append(button);
    }
    const strong = document.createElement("strong");
    strong.textContent = typeof value === "number" ? value.toLocaleString() : value;
    row.append(label, strong);
    return row;
  }));
}

function calculateMiniMax(model, tokens, sequences, tp, precision) {
  const indexerPrecision = Number($("indexer").value);
  const kvElementsPerToken = model.layers * 2 * model.heads * model.dim;
  const indexerElementsPerToken = model.sparseLayers * model.indexDim;
  const kvTotal = tokens * sequences * kvElementsPerToken * precision;
  const indexerTotal = tokens * sequences * indexerElementsPerToken * indexerPrecision;
  const perDevice = kvTotal / tp + indexerTotal;
  const total = perDevice * tp;
  return {
    kvTotal, sideTotal: indexerTotal, total,
    perDevice,
    label: "MiniMax MSA sparse attention",
    rows: [
      ["kv_bytes", "tokens × sequences × layers × 2 × num_key_value_heads × head_dim × kv_precision_bytes"],
      ["indexer_bytes", "tokens × sequences × sparse_attention_layers × index_head_dim × indexer_precision_bytes"],
      ["per_device_bytes", "kv_bytes / TP size + indexer_bytes"],
      ["all_device_bytes", "kv_bytes + TP size × indexer_bytes"]
    ],
    note: "MiniMax Sparse Attention (MSA) uses a lightweight indexer to select relevant KV blocks, with a separate key-only indexer cache for block selection.",
    metrics: [["Per-device KV cache", kvTotal / tp], ["Per-device Indexer cache", indexerTotal]],
    perTokenLabel: "Per-device per token size",
    perTokenValue: (kvTotal / tp + indexerTotal) / (tokens * sequences),
    breakdown: [
      ["Main layers", model.layers],
      ["Full-attention layers", model.fullLayers, "Dense/full-attention layers without the MSA indexer branch."],
      ["Sparse-attention layers", model.sparseLayers, "MSA layers that add a key-only indexer side cache."],
      ["Per-device KV elements per token", kvElementsPerToken / tp, "Main K/V elements per token on each TP rank before applying KV precision."],
      ["Per-device Indexer elements per token", indexerElementsPerToken, "The complete MSA index-key cache is stored on every TP rank."],
      ["Index Q heads", model.indexQHeads, "Indexer query heads used for scoring selected KV blocks."],
      ["Index K heads", model.indexKHeads, "The Indexer stores one key head, replicated across tensor-parallel ranks."],
      ["Index block size", model.indexBlockSize], ["Top-k blocks", model.indexTopkBlocks],
      ["Local blocks", model.indexLocalBlocks], ["MTP modules not included", model.mtpModules],
      ["Next-N layers not included", model.nextNLayers],
      ["KV precision bytes", precision], ["Indexer precision bytes", indexerPrecision]
    ]
  };
}

function calculateQwen(model, tokens, sequences, tp, precision) {
  const includeLinear = $("linear").checked;
  const kvElementsPerToken = model.fullLayers * 2 * model.heads * model.dim;
  const kvTotal = tokens * sequences * kvElementsPerToken * precision;
  const convElementsPerSequence = model.linearLayers * model.linearConvKernel
    * (2 * model.linearKeyHeads * model.linearKeyDim + model.linearValueHeads * model.linearValueDim);
  const recurrentElementsPerSequence = model.linearLayers * model.linearValueHeads
    * model.linearKeyDim * model.linearValueDim;
  const convStateTotal = includeLinear ? sequences * convElementsPerSequence * 2 : 0;
  const recurrentStateTotal = includeLinear ? sequences * recurrentElementsPerSequence * 4 : 0;
  const linearStateTotal = convStateTotal + recurrentStateTotal;
  const total = kvTotal + linearStateTotal;
  const rows = [
    ["full_kv_bytes", "tokens × sequences × full_attention_layers × 2 × num_key_value_heads × head_dim × precision_bytes"]
  ];
  if (includeLinear) {
    rows.push(
      ["linear_conv_state_bytes", "sequences × linear_attention_layers × kernel_dim × (2 × key_heads × key_dim + value_heads × value_dim) × 2"],
      ["linear_recurrent_state_bytes", "sequences × linear_attention_layers × value_heads × key_dim × value_dim × 4"],
      ["total_bytes", "full_kv_bytes + linear_conv_state_bytes + linear_recurrent_state_bytes"]
    );
  } else {
    rows.push(["linear_attention_state", "excluded unless Include linear-attention state is enabled"], ["total_bytes", "full_kv_bytes"]);
  }
  return {
    kvTotal, sideTotal: linearStateTotal, total, perDevice: total / tp,
    label: "Qwen linear/full hybrid",
    rows,
    note: includeLinear
      ? "Qwen3.6 linear-attention state is fixed per sequence rather than per token, so it matters more for short prompts."
      : "Qwen3.6 linear-attention recurrent and convolution state is excluded by default. Enable the option to add the fixed runtime-state estimate.",
    metrics: [["Per-device Full-attention KV", kvTotal / tp], ["Per-device Linear-attention state", linearStateTotal / tp]],
    perTokenLabel: "Per-device per token size",
    perTokenValue: total / (tokens * sequences * tp),
    breakdown: [
      ["Main layers", model.layers], ["Full-attention layers", model.fullLayers],
      ["Linear-attention layers", model.linearLayers], ["Linear state included", includeLinear ? "Yes" : "No"],
      ["Per-device linear conv elements", convElementsPerSequence / tp],
      ["Per-device linear recurrent elements", recurrentElementsPerSequence / tp],
      ["MTP layers not included", model.mtpLayers], ["Per-device KV elements per token", kvElementsPerToken / tp],
      ["KV precision bytes", precision]
    ]
  };
}

function calculate() {
  const model = selected();
  const tokens = Math.max(1, Number($("tokens").value) || 1);
  const sequences = Math.max(1, Number($("sequences").value) || 1);
  const tp = Math.max(1, Number($("tensor-parallel").value) || 1);
  const precision = Number($("precision").value);
  const result = model.type === "minimax_msa"
    ? calculateMiniMax(model, tokens, sequences, tp, precision)
    : calculateQwen(model, tokens, sequences, tp, precision);

  $("per-device-gib").textContent = `${(result.perDevice / 1024 ** 3).toFixed(5)} GiB`;
  $("per-device-gb").textContent = `= ${(result.perDevice / 1e9).toFixed(5)} GB`;
  $("total-cache-metric").textContent = formatBytes(result.total);
  $("metric-one-label").textContent = result.metrics[0][0];
  $("metric-one").textContent = formatBytes(result.metrics[0][1]);
  $("metric-two-label").textContent = result.metrics[1][0];
  $("metric-two").textContent = formatBytes(result.metrics[1][1]);
  $("metric-three-label").textContent = result.perTokenLabel || "Per token size";
  $("metric-three").textContent = formatBytes(result.perTokenValue || result.total / (tokens * sequences));
  $("formula-label").textContent = result.label;
  renderFormula(result.rows);
  $("cache-note").textContent = result.note;
  renderBreakdown([...result.breakdown, ["Tensor parallel size", tp], ["Per-device cache size", formatBytes(result.perDevice)]]);
  const sourceUrl = `https://huggingface.co/${model.source}/raw/main/config.json`;
  $("source").textContent = sourceUrl;
  $("source-link").href = sourceUrl;
}

$("family").addEventListener("change", fillModels);
$("model").addEventListener("change", syncModel);
$("cache-form").addEventListener("input", calculate);
fillModels();
