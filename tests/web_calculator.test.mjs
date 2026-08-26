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

test("includes all 52 upstream models, nine visible families, and seven formulas", () => {
  assert.equal(data.models.length, 52);
  assert.deepEqual(app.families(data.models), [
    "Cohere", "DeepSeek", "Gemma", "GLM", "Kimi", "Llama", "MiMo", "MiniMax", "Qwen",
  ]);
  assert.equal(new Set(data.models.map((item) => item.formula)).size, 7);
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

test("every model formula derives per-device bytes from per-sequence cache groups", () => {
  for (const item of data.models) {
    for (const tp of app.validTpSizes(item)) {
      for (const includeDraftKvCache of [false, true]) {
        for (const includeLinearAttentionState of [false, true]) {
          const view = app.calculateView(item, inputFor(item, {
            tensorParallel: tp,
            includeDraftKvCache,
            includeLinearAttentionState,
          }), data);
          const rows = app.formulaRowsForView(view);
          const byName = Object.fromEntries(rows.map((row) => [row.name, row.expression]));

          assert.match(byName.per_device_bytes, /^sequences × .+$/, item.id);
          if (byName.per_device_bytes.includes(" + ")) {
            assert.match(byName.per_device_bytes, /^sequences × \(.+\)$/, item.id);
          } else {
            assert.doesNotMatch(byName.per_device_bytes, /[()]/, item.id);
          }
          assert.equal(byName.all_device_bytes, "TP size × per_device_bytes", item.id);
          assert.equal(Object.hasOwn(byName, "total_bytes"), false, item.id);
          assert.equal(rows.some((row) => /total_bytes\s*\/\s*TP size/.test(row.expression)), false, item.id);
          assert.equal(
            rows.some((row) => row.name !== "per_device_bytes" && /\bsequences\b/.test(row.expression)),
            false,
            item.id,
          );
        }
      }
    }
  }
});

test("MiniMax M3 shards KV but replicates its one-key-head Indexer cache", () => {
  const item = model("minimax-m3");
  const result = app.calculateView(item, inputFor(item, { tensorParallel: 4 }), data);
  assert.equal(result.perDeviceBytes / 1024 ** 2, 44.25);
  assert.equal(result.allDeviceBytes / 1024 ** 2, 177);

  const indexer = result.deviceGroups.find((group) => group.role === "indexer");
  const kv = result.deviceGroups.find((group) => group.role === "kv");
  assert.equal(kv.heads, 4);
  assert.equal(kv.localHeads, 1);
  assert.equal(kv.replicated, false);
  assert.equal(indexer.perDeviceBytes, indexer.bytes);
  assert.equal(indexer.allDeviceBytes, indexer.bytes * 4);
  assert.equal(indexer.heads, 1);
  assert.equal(item.fields.index_num_key_heads, indexer.heads);
  assert.equal(indexer.localHeads, 1);
  assert.equal(indexer.replicated, true);
});

test("cache distribution depends on heads rather than cache role", () => {
  const oneHeadKv = core.distributeHeads(1, 4);
  const fourHeadIndexer = core.distributeHeads(4, 4);

  assert.deepEqual(oneHeadKv, {
    heads: 1,
    localHeads: 1,
    replicated: true,
    replicationFactor: 4,
    perDeviceRatio: 1,
  });
  assert.deepEqual(fourHeadIndexer, {
    heads: 4,
    localHeads: 1,
    replicated: false,
    replicationFactor: 1,
    perDeviceRatio: 0.25,
  });
});

test("one-head MLA KV replicates even though its role is kv", () => {
  const item = model("kimi-k2.5");
  const result = app.calculateView(item, inputFor(item, { tensorParallel: 4 }), data);
  const kv = result.deviceGroups.find((group) => group.role === "kv");

  assert.equal(kv.heads, 1);
  assert.equal(kv.localHeads, 1);
  assert.equal(kv.replicated, true);
  assert.equal(kv.perDeviceBytes, kv.bytes);
  assert.equal(kv.allDeviceBytes, kv.bytes * 4);
  const rows = Object.fromEntries(
    app.formulaRowsForView(result).map((row) => [row.name, row.expression]),
  );
  assert.equal(rows.per_device_bytes, "sequences × kv_bytes");
});

test("core API applies per-group head distribution at TP greater than one", () => {
  const item = model("minimax-m3");
  const result = core.calculate(item, {
    tokens: 1024,
    sequences: 1,
    precision: "bf16_fp16",
    indexerPrecision: "bf16_fp16",
    tensorParallel: 4,
  }, {
    precisionOptions: data.precision_options,
    indexerPrecisionOptions: data.indexer_precision_options,
  });
  const kv = result.cacheGroups.find((group) => group.role === "kv");
  const indexer = result.cacheGroups.find((group) => group.role === "indexer");

  assert.equal(kv.perDeviceBytes, kv.bytes / 4);
  assert.equal(indexer.perDeviceBytes, indexer.bytes);
  assert.equal(result.perDeviceBytes, kv.bytes / 4 + indexer.bytes);
  assert.equal(result.allDeviceBytes, kv.bytes + indexer.bytes * 4);
});

test("MiniMax M3 formula substitutes local heads inside each cache group", () => {
  const item = model("minimax-m3");
  const view = app.calculateView(item, inputFor(item, { tensorParallel: 4 }), data);
  const rows = Object.fromEntries(
    app.formulaRowsForView(view).map((row) => [row.name, row.expression]),
  );

  assert.match(rows.kv_bytes, /max\(num_key_value_heads \/ TP size, 1\)/);
  assert.match(rows.indexer_bytes, /max\(index_num_key_heads \/ TP size, 1\)/);
  assert.equal(rows.per_device_bytes, "sequences × (kv_bytes + indexer_bytes)");
  assert.equal(rows.all_device_bytes, "TP size × per_device_bytes");
  const details = Object.fromEntries(app.detailsForView(view).map((row) => [row[0], row[1]]));
  assert.equal(details["Index K heads"], item.fields.index_num_key_heads);
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

test("DeepSeek V4 formula keeps its one-head cache groups per TP device", () => {
  const item = model("deepseek-v4-pro");
  const view = app.calculateView(item, inputFor(item), data);
  const rows = Object.fromEntries(
    app.formulaRowsForView(view).map((row) => [row.name, row.expression]),
  );

  assert.match(rows.indexer_bytes, /max\(index_num_key_heads \/ TP size, 1\)/);
  assert.equal(rows.per_device_bytes, "sequences × (kv_bytes + indexer_bytes)");
  assert.equal(rows.all_device_bytes, "TP size × per_device_bytes");
  assert.equal(Object.hasOwn(rows, "total_bytes"), false);
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
