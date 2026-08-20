import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const data = require("../web/assets/models.js");
const core = require("../web/assets/calculator-core.js");
const app = require("../web/assets/script.js");

function model(id) {
  return data.models.find((item) => item.id === id);
}

function inputFor(item, overrides = {}) {
  return {
    tokens: item.default_tokens || 1024,
    sequences: 1,
    tensorParallel: 1,
    precision: app.defaultPrecision(item),
    indexerPrecision: app.defaultIndexerPrecision(item),
    includeDraftKvCache: false,
    includeLinearAttentionState: app.hasLinearState(item),
    ...overrides,
  };
}

test("includes all 53 upstream models, nine visible families, and eight formulas", () => {
  assert.equal(data.models.length, 53);
  assert.deepEqual(app.families(data.models), [
    "Cohere", "DeepSeek", "Gemma", "GLM", "Kimi", "Llama", "MiMo", "MiniMax", "Qwen",
  ]);
  assert.equal(new Set(data.models.map((item) => item.formula)).size, 8);
  assert.equal(app.modelsForFamily(data.models, "Qwen").length, 23);
});

test("every model calculates finite values at every supported TP size", () => {
  for (const item of data.models) {
    for (const tp of app.validTpSizes(item)) {
      const result = app.calculateView(item, inputFor(item, { tensorParallel: tp }), data);
      assert.ok(Number.isFinite(result.perDeviceBytes) && result.perDeviceBytes > 0, item.id);
      assert.ok(Number.isFinite(result.allDeviceBytes) && result.allDeviceBytes > 0, item.id);
      assert.equal(result.allDeviceBytes, result.perDeviceBytes * tp, item.id);
      assert.equal(item.fields.num_key_value_heads % tp, 0, item.id);
    }
  }
});

test("MiniMax M3 shards KV but replicates its one-key-head Indexer cache", () => {
  const item = model("minimax-m3");
  const result = app.calculateView(item, inputFor(item, { tensorParallel: 4 }), data);
  assert.equal(result.perDeviceBytes / 1024 ** 2, 44.25);
  assert.equal(result.allDeviceBytes / 1024 ** 2, 177);

  const indexer = result.deviceGroups.find((group) => group.role === "indexer");
  assert.equal(indexer.perDeviceBytes, indexer.bytes);
  assert.equal(indexer.allDeviceBytes, indexer.bytes * 4);
});

test("MiniMax M3 accepts FP8/INT8 Indexer precision", () => {
  const item = model("minimax-m3");
  const result = app.calculateView(item, inputFor(item, {
    tensorParallel: 4,
    indexerPrecision: "fp8_int8",
  }), data);
  assert.equal(result.indexerPrecisionLabel, "FP8 / INT8");
  assert.equal(result.perDeviceBytes / 1024 ** 2, 37.125);
});

test("Qwen3.6-27B keeps linear-attention state enabled and sharded per device", () => {
  const item = model("qwen3.6-27b");
  const result = app.calculateView(item, inputFor(item, { tensorParallel: 4 }), data);
  assert.equal(result.perDeviceBytes / 1024 ** 2, 52.9375);
  assert.equal(result.allDeviceBytes / 1024 ** 2, 211.75);
  assert.equal(
    result.elementPlan.components.find(([label]) => label === "Linear state included")[1],
    "Yes",
  );
});

test("Kimi K3 combines token-linear MLA KV with constant KDA state", () => {
  const item = model("kimi-k3");
  assert.equal(app.hasLinearState(item), true);
  assert.equal(app.hasKdaSpecTokens(item), true);
  const result = app.calculateView(item, inputFor(item), data);
  // MLA latent: 24 full-attention layers x (512 + 64) elements per token.
  assert.equal(result.elementPlan.elementsPerToken, 24 * 576);
  // bf16: KV 28311552 + conv 15261696 + recurrent 434110464 bytes per sequence.
  assert.equal(result.allDeviceBytes, 477683712);
  const mla = result.deviceGroups.find((group) => group.role === "kv");
  assert.equal(mla.replicated, true);
  assert.equal(mla.bytes, 28311552);
});

test("Kimi K3 replicates MLA latent but shards KDA state across TP", () => {
  const item = model("kimi-k3");
  const result = app.calculateView(item, inputFor(item, { tensorParallel: 8 }), data);
  // Per device: full MLA copy + (conv + recurrent) / 8.
  assert.equal(result.perDeviceBytes, 28311552 + (15261696 + 434110464) / 8);
  const kdaGroups = result.deviceGroups.filter((group) => group.role === "linear_state");
  assert.equal(kdaGroups.length, 2);
  kdaGroups.forEach((group) => {
    assert.equal(group.replicated, false);
  });
});

test("Kimi K3 speculative tokens widen the KDA conv-state window", () => {
  const item = model("kimi-k3");
  const base = app.calculateView(item, inputFor(item), data);
  const spec = app.calculateView(item, inputFor(item, { kdaSpecTokens: 1 }), data);
  // One extra slot: 69 layers x (3 x 96 x 128) elements x 2 bytes.
  assert.equal(spec.allDeviceBytes - base.allDeviceBytes, 69 * 3 * 96 * 128 * 2);
  assert.equal(spec.allDeviceBytes, 482770944);
});

test("Kimi K3 excludes KDA state when the linear-attention option is off", () => {
  const item = model("kimi-k3");
  const result = app.calculateView(
    item,
    inputFor(item, { includeLinearAttentionState: false }),
    data,
  );
  assert.equal(result.allDeviceBytes, 28311552);
  assert.equal(
    result.elementPlan.components.find(([label]) => label === "Linear state included")[1],
    "No",
  );
});

test("TP formula rows name the actual sharded and replicated caches", () => {
  const k3 = app.calculateView(
    model("kimi-k3"),
    inputFor(model("kimi-k3"), { tensorParallel: 8 }),
    data,
  );
  const k3PerDevice = app
    .formulaRowsForView(k3)
    .find((row) => row.name === "per_device_bytes");
  assert.equal(
    k3PerDevice.expression,
    "(kda_conv_state_bytes + kda_recurrent_state_bytes) / TP size + mla_kv_bytes",
  );
  assert.match(k3PerDevice.description, /MLA KV cache/);
  assert.match(k3PerDevice.description, /KDA conv state, KDA recurrent state/);

  const m3 = app.calculateView(
    model("minimax-m3"),
    inputFor(model("minimax-m3"), { tensorParallel: 4 }),
    data,
  );
  const m3PerDevice = app
    .formulaRowsForView(m3)
    .find((row) => row.name === "per_device_bytes");
  assert.equal(m3PerDevice.expression, "(kv_bytes) / TP size + indexer_bytes");
});

test("token presets include 128K and the model's maximum context", () => {
  const k3 = app.tokenPresets(model("kimi-k3"));
  assert.deepEqual(
    k3.filter((preset) => preset.value === 131072).map((preset) => preset.label),
    ["128K"],
  );
  assert.deepEqual(k3[k3.length - 1], { value: 1048576, label: "1M (max)" });
  const small = app.tokenPresets(model("qwen3.6-27b"));
  assert.ok(small.every((preset) => preset.value <= 262144));
});

test("standard GQA result matches the upstream Qwen3-32B golden value", () => {
  const item = model("qwen3-32b");
  const result = core.calculate(item, {
    tokens: 128000,
    sequences: 1,
    precision: "bf16_fp16",
    tensorParallel: 1,
  }, {
    precisionOptions: data.precision_options,
    indexerPrecisionOptions: data.indexer_precision_options,
  });
  assert.equal(result.elementPlan.elementsPerToken, 131072);
  assert.equal(result.totalGiB, 31.25);
});

test("DeepSeek V4 defaults to FP8 KV and FP4 Indexer cache", () => {
  const item = model("deepseek-v4-pro");
  assert.equal(app.defaultPrecision(item), "fp8_int8");
  assert.equal(app.defaultIndexerPrecision(item), "fp4_int4");
  const result = app.calculateView(item, inputFor(item), data);
  assert.equal(result.precisionLabel, "FP8 / INT8");
  assert.equal(result.indexerPrecisionLabel, "FP4 / INT4");
});

test("invalid TP size is rejected", () => {
  const item = model("qwen3.6-27b");
  assert.throws(
    () => app.calculateView(item, inputFor(item, { tensorParallel: 3 }), data),
    /num_key_value_heads \/ TP must be an integer/,
  );
});

test("TP help explains why only KV-head divisors are available", () => {
  const item = model("minimax-m3");
  assert.deepEqual(app.validTpSizes(item), [1, 2, 4]);
  assert.equal(
    app.tpHelpText(item),
    "This model has 4 KV heads, so TP size can be 1, 2, or 4. Only divisors are shown because num_key_value_heads / TP size must be an integer, giving each device a whole number of KV heads.",
  );
});

test("legacy numeric precision values do not produce NaN", () => {
  const item = model("minimax-m3");
  const result = app.calculateView(item, inputFor(item, {
    precision: "2",
    indexerPrecision: "1",
  }), data);
  assert.equal(result.precisionLabel, "BF16 / FP16");
  assert.equal(result.indexerPrecisionLabel, "FP8 / INT8");
  assert.ok(Number.isFinite(result.perDeviceBytes));
});
