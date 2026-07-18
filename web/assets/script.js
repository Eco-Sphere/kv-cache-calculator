(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(
      require("./calculator-core.js"),
      require("./models.js"),
      null
    );
    return;
  }

  const app = factory(root.KVCacheCalculator, root.KV_MODEL_DATA, root.document);
  root.KVCacheCalculatorApp = app;
  const start = function () {
    app.mount(root.document, root.KV_MODEL_DATA);
  };
  if (root.document.readyState === "loading") {
    root.document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function (Core, MODEL_DATA, document) {
  "use strict";

  if (!Core || !MODEL_DATA || !Array.isArray(MODEL_DATA.models)) {
    throw new Error("KV cache calculator data failed to load.");
  }

  const BYTES_PER_GB = 1e9;
  const BYTES_PER_GIB = 1024 ** 3;

  function numericField(model, name, fallback) {
    const value = Number(model && model.fields && model.fields[name]);
    return Number.isFinite(value) ? value : fallback;
  }

  function visibleFamily(model) {
    return Core.modelFamily(model);
  }

  function families(models) {
    return Array.from(new Set(models.map(visibleFamily))).sort(function (a, b) {
      return a.localeCompare(b);
    });
  }

  function modelsForFamily(models, family) {
    return models.filter(function (model) {
      return visibleFamily(model) === family;
    });
  }

  function validTpSizes(model) {
    const kvHeads = Math.max(1, Math.floor(numericField(model, "num_key_value_heads", 1)));
    return Array.from({ length: kvHeads }, function (_, index) {
      return index + 1;
    }).filter(function (tp) {
      return kvHeads % tp === 0;
    });
  }

  function hasIndexer(model) {
    return Number.isFinite(numericField(model, "index_head_dim", NaN));
  }

  function draftLayerCount(model) {
    if (!model || !model.fields || model.fields.disable_draft_kv_cache === true) {
      return 0;
    }
    const nextN = numericField(model, "num_nextn_predict_layers", 0);
    if (nextN > 0) {
      return nextN;
    }
    if (model.fields.use_mtp === true) {
      return numericField(model, "num_mtp_modules", 0)
        * numericField(model, "mtp_transformer_layers", 0);
    }
    return 0;
  }

  function hasDraftCache(model) {
    if (model.formula === "deepseek_v4_hybrid") {
      const mainLayers = numericField(model, "num_hidden_layers", 0);
      return Array.isArray(model.fields.compress_ratios)
        && model.fields.compress_ratios.length > mainLayers;
    }
    return draftLayerCount(model) > 0;
  }

  function hasLinearState(model) {
    return model.formula === "qwen_linear_full_hybrid";
  }

  function defaultPrecision(model) {
    return model.formula === "deepseek_v4_hybrid" ? "fp8_int8" : "bf16_fp16";
  }

  function defaultIndexerPrecision(model) {
    return model.formula === "deepseek_v4_hybrid" ? "fp4_int4" : "bf16_fp16";
  }

  function modelForCalculation(model) {
    if (model.id !== "minimax-m3") {
      return model;
    }
    const fields = Object.assign({}, model.fields);
    delete fields.indexer_fixed_precision_id;
    return Object.assign({}, model, { fields: fields });
  }

  function precisionConfig(data) {
    return {
      precisionOptions: data.precision_options,
      indexerPrecisionOptions: data.indexer_precision_options
    };
  }

  function calculateView(model, input, data) {
    const sourceData = data || MODEL_DATA;
    const requestedTp = Math.max(1, Math.floor(Number(input.tensorParallel) || 1));
    const options = validTpSizes(model);
    if (!options.includes(requestedTp)) {
      throw new RangeError(
        "TP size " + requestedTp + " is invalid because num_key_value_heads / TP must be an integer."
      );
    }

    const result = Core.calculate(
      modelForCalculation(model),
      {
        tokens: input.tokens,
        sequences: input.sequences,
        precision: input.precision || defaultPrecision(model),
        indexerPrecision: input.indexerPrecision || defaultIndexerPrecision(model),
        includeDraftKvCache: Boolean(input.includeDraftKvCache),
        includeLinearAttentionState: Boolean(input.includeLinearAttentionState),
        tensorParallel: 1
      },
      precisionConfig(sourceData)
    );

    const deviceGroups = result.cacheGroups.map(function (group) {
      const replicated = group.role === "indexer";
      const perDeviceBytes = replicated ? group.bytes : group.bytes / requestedTp;
      return Object.assign({}, group, {
        replicated: replicated,
        perDeviceBytes: perDeviceBytes,
        allDeviceBytes: perDeviceBytes * requestedTp
      });
    });
    const perDeviceBytes = deviceGroups.reduce(function (sum, group) {
      return sum + group.perDeviceBytes;
    }, 0);
    const allDeviceBytes = deviceGroups.reduce(function (sum, group) {
      return sum + group.allDeviceBytes;
    }, 0);

    return Object.assign({}, result, {
      model: model,
      tensorParallel: requestedTp,
      logicalBytes: result.totalBytes,
      perDeviceBytes: perDeviceBytes,
      perDeviceGiB: perDeviceBytes / BYTES_PER_GIB,
      perDeviceGB: perDeviceBytes / BYTES_PER_GB,
      allDeviceBytes: allDeviceBytes,
      deviceGroups: deviceGroups,
      perDeviceBytesPerSequence: perDeviceBytes / result.sequences,
      perDeviceBytesPerToken: perDeviceBytes / (result.tokens * result.sequences)
    });
  }

  function formatNumber(value) {
    if (typeof value !== "number") {
      return value;
    }
    return value.toLocaleString(undefined, {
      maximumFractionDigits: Number.isInteger(value) ? 0 : 2
    });
  }

  function scaledDetail(label, value, help, divisor, nextLabel) {
    return [
      nextLabel || label,
      typeof value === "number" ? value / divisor : value,
      help
    ];
  }

  function detailsForView(view) {
    const model = view.model;
    const formula = model.formula;
    const tp = view.tensorParallel;
    const original = view.components.map(function (row) {
      return row.slice();
    });
    const values = Object.fromEntries(original.map(function (row) {
      return [row[0], row[1]];
    }));
    const rows = [];

    original.forEach(function (row) {
      const label = row[0];
      const value = row[1];
      const help = row[2];

      if (formula === "minimax_msa" && label === "Index heads") {
        rows.push(["Index Q heads", numericField(model, "index_n_heads", 4), help]);
        rows.push([
          "Index K heads",
          1,
          "The Indexer stores one key head, replicated across tensor-parallel devices."
        ]);
        return;
      }

      if ((formula === "standard_gqa" || formula === "mla") && label === "Per-token elements") {
        rows.push(scaledDetail(
          label,
          value,
          help,
          tp,
          "Per-device KV elements per token"
        ));
        return;
      }

      if (formula === "dsa_mla") {
        if (label === "KV elements per token") {
          rows.push(scaledDetail(label, value, help, tp, "Per-device KV elements per token"));
          return;
        }
        if (label === "Indexer elements per token") {
          rows.push(["Per-device Indexer elements per token", value, help]);
          return;
        }
        if (label === "Per-token elements") {
          const kv = Number(values["KV elements per token"]) || 0;
          const indexer = Number(values["Indexer elements per token"]) || 0;
          rows.push([
            "Per-device total elements per token",
            kv / tp + indexer,
            help
          ]);
          return;
        }
      }

      if (formula === "qwen_linear_full_hybrid") {
        if (label === "Linear conv elements") {
          rows.push(scaledDetail(label, value, help, tp, "Per-device linear conv elements"));
          return;
        }
        if (label === "Linear recurrent elements") {
          rows.push(scaledDetail(label, value, help, tp, "Per-device linear recurrent elements"));
          return;
        }
        if (label === "Per-token elements") {
          rows.push(scaledDetail(
            label,
            value,
            help,
            tp,
            "Per-device KV elements per token"
          ));
          return;
        }
      }

      if (formula === "mixed_full_sliding_gqa") {
        if (label === "Full-attention elements") {
          rows.push(scaledDetail(label, value, help, tp, "Per-device full-attention elements"));
          return;
        }
        if (label === "Sliding-window elements") {
          rows.push(scaledDetail(label, value, help, tp, "Per-device sliding-window elements"));
          return;
        }
      }

      if (formula === "minimax_msa") {
        if (label === "KV elements per token") {
          rows.push(scaledDetail(label, value, help, tp, "Per-device KV elements per token"));
          return;
        }
        if (label === "Indexer elements per token") {
          rows.push(["Per-device Indexer elements per token", value, help]);
          return;
        }
      }

      if (formula === "deepseek_v4_hybrid") {
        const shardedLabels = [
          "Ratio=0 KV elements",
          "Sliding-window elements",
          "Compressed elements",
          "KV elements"
        ];
        if (shardedLabels.includes(label)) {
          rows.push(scaledDetail(label, value, help, tp, "Per-device " + label.toLowerCase()));
          return;
        }
        if (label === "Indexer elements") {
          rows.push(["Per-device Indexer elements", value, help]);
          return;
        }
      }

      rows.push(row);
    });

    rows.push(["Tensor parallel size", tp]);
    rows.push(["Per-device cache size", Core.formatBytes(view.perDeviceBytes)]);
    return rows;
  }

  function formulaRowsForView(view) {
    const rows = (view.elementPlan.formulaRows || []).map(function (row) {
      return {
        name: row.name,
        expression: row.expression,
        description: row.description
      };
    });
    const hasReplicatedIndexer = view.deviceGroups.some(function (group) {
      return group.replicated;
    });
    if (hasReplicatedIndexer) {
      rows.push({
        name: "per_device_bytes",
        expression: "sharded_cache_bytes / TP size + replicated_indexer_bytes",
        description: "The indexer has one stored key head and is replicated on every TP device."
      });
      rows.push({
        name: "all_device_bytes",
        expression: "sharded_cache_bytes + TP size × replicated_indexer_bytes",
        description: "Physical cache across all devices includes one indexer copy per TP device."
      });
    } else {
      rows.push({
        name: "per_device_bytes",
        expression: "total_bytes / TP size",
        description: "The cache is evenly sharded across valid tensor-parallel devices."
      });
      rows.push({
        name: "all_device_bytes",
        expression: "total_bytes",
        description: "Sharded cache is counted once across all tensor-parallel devices."
      });
    }
    return rows;
  }

  function metricRowsForView(view) {
    const groups = view.deviceGroups;
    const first = groups[0]
      ? ["Per-device " + groups[0].label, groups[0].perDeviceBytes]
      : ["Per-device cache", view.perDeviceBytes];
    const second = groups[1]
      ? ["Per-device " + groups[1].label, groups[1].perDeviceBytes]
      : ["Per-device per sequence size", view.perDeviceBytesPerSequence];
    return [
      first,
      second,
      ["Per-device per token size", view.perDeviceBytesPerToken]
    ];
  }

  function appendHelp(label, help, doc) {
    if (!help) {
      return;
    }
    const button = doc.createElement("button");
    button.type = "button";
    button.className = "kv-help";
    button.textContent = "?";
    button.setAttribute("aria-label", help);
    button.dataset.tooltip = help;
    label.append(button);
  }

  function renderFormula(container, rows, doc) {
    container.replaceChildren.apply(container, rows.map(function (item) {
      const row = doc.createElement("div");
      row.className = "formula-row";
      const name = doc.createElement("span");
      name.className = "formula-name";
      name.textContent = item.name;
      appendHelp(name, item.description, doc);
      const equals = doc.createElement("span");
      equals.className = "formula-equals";
      equals.textContent = "=";
      const expression = doc.createElement("span");
      expression.className = "formula-expression";
      expression.textContent = item.expression;
      row.append(name, equals, expression);
      return row;
    }));
  }

  function renderBreakdown(container, rows, doc) {
    container.replaceChildren.apply(container, rows.map(function (item) {
      const row = doc.createElement("div");
      row.className = "breakdown-row";
      const label = doc.createElement("span");
      label.textContent = item[0];
      appendHelp(label, item[2], doc);
      const value = doc.createElement("strong");
      value.textContent = formatNumber(item[1]);
      row.append(label, value);
      return row;
    }));
  }

  function setOptions(select, options, value, doc) {
    select.replaceChildren.apply(select, options.map(function (optionData) {
      const option = doc.createElement("option");
      option.value = optionData.value;
      option.textContent = optionData.label;
      return option;
    }));
    if (options.some(function (optionData) { return optionData.value === value; })) {
      select.value = value;
    }
  }

  function mount(doc, data) {
    if (!doc) {
      return;
    }
    const sourceData = data || MODEL_DATA;
    const get = function (id) {
      return doc.getElementById(id);
    };
    const form = get("cache-form");
    if (!form) {
      return;
    }

    const controls = {
      family: get("family"),
      model: get("model"),
      tokens: get("tokens"),
      sequences: get("sequences"),
      tensorParallel: get("tensor-parallel"),
      precision: get("precision"),
      indexerPrecision: get("indexer"),
      indexerControl: get("indexer-control"),
      draft: get("draft"),
      draftControl: get("draft-control"),
      linear: get("linear"),
      linearControl: get("linear-control")
    };

    setOptions(
      controls.precision,
      sourceData.precision_options.map(function (option) {
        return { value: option.id, label: option.label };
      }),
      "bf16_fp16",
      doc
    );
    setOptions(
      controls.indexerPrecision,
      sourceData.indexer_precision_options.map(function (option) {
        return { value: option.id, label: option.label };
      }),
      "bf16_fp16",
      doc
    );

    const familyOptions = families(sourceData.models).map(function (family) {
      return { value: family, label: family };
    });
    setOptions(controls.family, familyOptions, "MiniMax", doc);

    function selectedModel() {
      return sourceData.models.find(function (model) {
        return model.id === controls.model.value;
      }) || modelsForFamily(sourceData.models, controls.family.value)[0];
    }

    function fillModels(preferredId) {
      const models = modelsForFamily(sourceData.models, controls.family.value);
      setOptions(
        controls.model,
        models.map(function (model) {
          return { value: model.id, label: model.label };
        }),
        preferredId,
        doc
      );
    }

    function render() {
      try {
        const model = selectedModel();
        const view = calculateView(model, {
          tokens: controls.tokens.value,
          sequences: controls.sequences.value,
          tensorParallel: controls.tensorParallel.value,
          precision: controls.precision.value,
          indexerPrecision: controls.indexerPrecision.value,
          includeDraftKvCache: controls.draft.checked,
          includeLinearAttentionState: controls.linear.checked
        }, sourceData);

        get("per-device-gib").textContent = view.perDeviceGiB.toFixed(5) + " GiB";
        get("per-device-gb").textContent = "= " + view.perDeviceGB.toFixed(5) + " GB";
        get("total-cache-metric").textContent = Core.formatBytes(view.allDeviceBytes);

        const metrics = metricRowsForView(view);
        get("metric-one-label").textContent = metrics[0][0];
        get("metric-one").textContent = Core.formatBytes(metrics[0][1]);
        get("metric-two-label").textContent = metrics[1][0];
        get("metric-two").textContent = Core.formatBytes(metrics[1][1]);
        get("metric-three-label").textContent = metrics[2][0];
        get("metric-three").textContent = Core.formatBytes(metrics[2][1]);

        get("formula-label").textContent = view.elementPlan.formulaLabel;
        renderFormula(get("formula-list"), formulaRowsForView(view), doc);
        const tpNote = view.deviceGroups.some(function (group) { return group.replicated; })
          ? " Per-device values shard non-indexer cache across TP and replicate the one-key-head indexer cache on every device."
          : " Per-device values assume even cache sharding across valid TP devices.";
        get("cache-note").textContent = view.elementPlan.note + tpNote;
        renderBreakdown(get("breakdown"), detailsForView(view), doc);

        get("source").textContent = model.source_url;
        get("source-link").href = model.source_url;
      } catch (error) {
        get("cache-note").textContent = error.message;
      }
    }

    function syncModel() {
      const model = selectedModel();
      controls.tokens.value = String(model.default_tokens || 1024);
      controls.tokens.max = String(model.max_position_embeddings || "");

      const tpOptions = validTpSizes(model).map(function (tp) {
        return { value: String(tp), label: String(tp) };
      });
      setOptions(controls.tensorParallel, tpOptions, "1", doc);
      controls.precision.value = defaultPrecision(model);
      controls.indexerPrecision.value = defaultIndexerPrecision(model);

      controls.indexerControl.hidden = !hasIndexer(model);
      controls.draftControl.hidden = !hasDraftCache(model);
      controls.draft.checked = false;
      controls.linearControl.hidden = !hasLinearState(model);
      controls.linear.checked = hasLinearState(model);
      render();
    }

    controls.family.addEventListener("change", function () {
      fillModels();
      syncModel();
    });
    controls.model.addEventListener("change", syncModel);
    form.addEventListener("input", function (event) {
      if (event.target !== controls.family && event.target !== controls.model) {
        render();
      }
    });

    fillModels("minimax-m3");
    syncModel();
  }

  return {
    calculateView: calculateView,
    defaultIndexerPrecision: defaultIndexerPrecision,
    defaultPrecision: defaultPrecision,
    detailsForView: detailsForView,
    families: families,
    hasDraftCache: hasDraftCache,
    hasIndexer: hasIndexer,
    hasLinearState: hasLinearState,
    metricRowsForView: metricRowsForView,
    modelsForFamily: modelsForFamily,
    mount: mount,
    validTpSizes: validTpSizes
  };
});
