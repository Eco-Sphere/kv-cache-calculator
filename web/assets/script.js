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
  const PRECISION_ALIASES = {
    "2": "bf16_fp16",
    "1": "fp8_int8",
    "0.5": "fp4_int4"
  };

  function normalizePrecision(value, fallback) {
    const raw = String(value || fallback);
    return PRECISION_ALIASES[raw] || raw;
  }

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

  function tpHelpText(model) {
    const kvHeads = Math.max(1, Math.floor(numericField(model, "num_key_value_heads", 1)));
    const sizes = validTpSizes(model);
    const choices = sizes.length === 1
      ? String(sizes[0])
      : sizes.length === 2
        ? sizes.join(" or ")
        : sizes.slice(0, -1).join(", ") + ", or " + sizes[sizes.length - 1];
    const headLabel = kvHeads === 1 ? "KV head" : "KV heads";
    return "This model has " + kvHeads + " " + headLabel + ", so TP size can be " + choices
      + ", because num_key_value_heads / TP size must be an integer.";
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

  function speculativeModelsFor(data, model) {
    const ids = (model && model.speculative_model_ids) || [];
    const pool = (data && data.speculative_models) || [];
    return ids
      .map(function (id) {
        return pool.find(function (item) {
          return item.id === id;
        });
      })
      .filter(Boolean);
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
    const specModel = input.speculativeModel || null;
    if (specModel) {
      const draftKvHeads = Math.max(1, Math.floor(numericField(specModel, "num_key_value_heads", 1)));
      if (draftKvHeads % requestedTp !== 0) {
        throw new RangeError(
          "TP size " + requestedTp + " is invalid for speculative model " + specModel.label
            + " because its num_key_value_heads / TP must be an integer."
        );
      }
    }

    const result = Core.calculate(
      modelForCalculation(model),
      {
        tokens: input.tokens,
        sequences: input.sequences,
        precision: normalizePrecision(input.precision, defaultPrecision(model)),
        indexerPrecision: normalizePrecision(
          input.indexerPrecision,
          defaultIndexerPrecision(model)
        ),
        includeDraftKvCache: Boolean(input.includeDraftKvCache),
        includeLinearAttentionState: Boolean(input.includeLinearAttentionState),
        specTokens: input.specTokens,
        speculativeModel: specModel,
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
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", "kv-live-tooltip");
    button.dataset.tooltip = help;
    label.append(button);
  }

  function bindHelpTooltips(doc) {
    if (!doc || !doc.body) {
      return;
    }
    const win = doc.defaultView;
    if (!win) {
      return;
    }

    doc.documentElement.classList.add("kv-tooltips-ready");

    let tip = doc.getElementById("kv-live-tooltip");
    if (!tip) {
      tip = doc.createElement("div");
      tip.id = "kv-live-tooltip";
      tip.className = "kv-live-tooltip";
      tip.setAttribute("aria-hidden", "true");
      tip.hidden = true;
      doc.body.appendChild(tip);
    }

    let active = null;
    let sticky = false;

    function hide() {
      if (active) {
        active.setAttribute("aria-expanded", "false");
      }
      active = null;
      sticky = false;
      tip.hidden = true;
    }

    function place() {
      if (active && !active.isConnected) {
        hide();
        return;
      }
      if (!active || tip.hidden) {
        return;
      }
      const rect = active.getBoundingClientRect();
      const gap = 8;
      const pad = 8;
      const maxWidth = Math.min(320, win.innerWidth - pad * 2);
      tip.style.maxWidth = maxWidth + "px";
      const width = tip.offsetWidth;
      const height = tip.offsetHeight;
      let left = rect.left + rect.width / 2 - width / 2;
      left = Math.min(Math.max(pad, left), win.innerWidth - width - pad);
      let top = rect.top - height - gap;
      if (top < pad) {
        top = rect.bottom + gap;
        if (top + height > win.innerHeight - pad) {
          top = Math.max(pad, win.innerHeight - height - pad);
        }
      }
      tip.style.left = Math.round(left) + "px";
      tip.style.top = Math.round(top) + "px";
    }

    function show(button, makeSticky) {
      const text = button.getAttribute("data-tooltip") || button.getAttribute("aria-label") || "";
      if (!text) {
        return;
      }
      if (active && active !== button) {
        active.setAttribute("aria-expanded", "false");
      }
      active = button;
      sticky = Boolean(makeSticky);
      tip.textContent = text;
      tip.hidden = false;
      button.setAttribute("aria-expanded", "true");
      place();
    }

    doc.addEventListener("pointerover", function (event) {
      const button = event.target.closest && event.target.closest(".kv-help");
      if (!button || button === active) {
        return;
      }
      show(button, false);
    });
    doc.addEventListener("pointerout", function (event) {
      const button = event.target.closest && event.target.closest(".kv-help");
      if (!button || sticky || active !== button) {
        return;
      }
      hide();
    });
    doc.addEventListener("focusin", function (event) {
      const button = event.target.closest && event.target.closest(".kv-help");
      if (button) {
        show(button, true);
      }
    });
    doc.addEventListener("focusout", function (event) {
      const button = event.target.closest && event.target.closest(".kv-help");
      if (!button || active !== button) {
        return;
      }
      hide();
    });
    doc.addEventListener("click", function (event) {
      const button = event.target.closest && event.target.closest(".kv-help");
      if (button) {
        if (active === button && sticky) {
          hide();
        } else {
          show(button, true);
        }
        return;
      }
      hide();
    });
    doc.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        hide();
      }
    });
    win.addEventListener("scroll", function () {
      if (active) {
        place();
      }
    }, true);
    win.addEventListener("resize", function () {
      if (active) {
        place();
      }
    });
    doc.addEventListener("kv-help-refresh", function () {
      if (active && !active.isConnected) {
        hide();
      }
    });
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

    bindHelpTooltips(doc);

    const controls = {
      family: get("family"),
      model: get("model"),
      tokens: get("tokens"),
      sequences: get("sequences"),
      tensorParallel: get("tensor-parallel"),
      tpHelp: get("tp-help"),
      tpHelpDescription: get("tp-help-description"),
      precision: get("precision"),
      indexerPrecision: get("indexer"),
      indexerControl: get("indexer-control"),
      draft: get("draft"),
      draftControl: get("draft-control"),
      linear: get("linear"),
      linearControl: get("linear-control"),
      specModel: get("spec-model"),
      specModelControl: get("spec-model-control"),
      specTokens: get("spec-tokens"),
      specTokensControl: get("spec-tokens-control")
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
        const specModel = speculativeModelsFor(sourceData, model).find(function (item) {
          return item.id === controls.specModel.value;
        }) || null;
        controls.specTokensControl.hidden = !(specModel && hasLinearState(model));
        const view = calculateView(model, {
          tokens: controls.tokens.value,
          sequences: controls.sequences.value,
          tensorParallel: controls.tensorParallel.value,
          precision: controls.precision.value,
          indexerPrecision: controls.indexerPrecision.value,
          includeDraftKvCache: controls.draft.checked,
          includeLinearAttentionState: controls.linear.checked,
          specTokens: specModel ? controls.specTokens.value : 0,
          speculativeModel: specModel
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

        const draftDetails = draftDetailsForView(view);
        const draftPanel = get("draft-panel");
        draftPanel.hidden = !draftDetails;
        if (draftDetails) {
          get("draft-name").textContent = draftDetails.label;
          const draftLink = get("draft-link");
          draftLink.href = draftDetails.href || "#";
          draftLink.title = draftDetails.href || "";
          renderBreakdown(get("draft-breakdown"), draftDetails.rows, doc);
        }

        get("source").textContent = model.source_url;
        get("source-link").href = model.source_url;
      } catch (error) {
        get("cache-note").textContent = error.message;
        get("draft-panel").hidden = true;
      }
      if (typeof doc.dispatchEvent === "function") {
        doc.dispatchEvent(new Event("kv-help-refresh"));
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
      const tpDescription = tpHelpText(model);
      controls.tpHelp.dataset.tooltip = tpDescription;
      controls.tpHelpDescription.textContent = tpDescription;
      controls.tpHelp.setAttribute("aria-expanded", "false");
      controls.precision.value = defaultPrecision(model);
      controls.indexerPrecision.value = defaultIndexerPrecision(model);

      controls.indexerControl.hidden = !hasIndexer(model);
      controls.draftControl.hidden = !hasDraftCache(model);
      controls.draft.checked = false;
      controls.linearControl.hidden = !hasLinearState(model);
      controls.linear.checked = hasLinearState(model);

      const specOptions = speculativeModelsFor(sourceData, model);
      setOptions(
        controls.specModel,
        [{ value: "", label: "None" }].concat(specOptions.map(function (item) {
          return { value: item.id, label: item.repo_id || item.label };
        })),
        "",
        doc
      );
      controls.specModelControl.hidden = specOptions.length === 0;
      controls.specTokensControl.hidden = true;
      controls.specTokens.value = "0";
      render();
    }

    controls.family.addEventListener("change", function () {
      fillModels();
      syncModel();
    });
    controls.model.addEventListener("change", syncModel);
    controls.specModel.addEventListener("change", function () {
      const model = selectedModel();
      const selected = speculativeModelsFor(sourceData, model).find(function (item) {
        return item.id === controls.specModel.value;
      });
      // N drives the target model GDN state growth; use the serving default when provided.
      controls.specTokens.value = selected
        ? String(numericField(
          selected,
          "default_speculative_tokens",
          numericField(selected, "dflash_block_size", 1)
        ))
        : "0";
      render();
    });
    form.addEventListener("input", function (event) {
      if (event.target !== controls.family && event.target !== controls.model) {
        render();
      }
    });

    fillModels("minimax-m3");
    syncModel();
  }

  function draftDetailsForView(view) {
    const spec = view.elementPlan && view.elementPlan.speculative;
    if (!spec) {
      return null;
    }
    const tp = view.tensorParallel;
    return {
      label: spec.label,
      href: spec.href,
      description: spec.description,
      rows: spec.components.map(function (row) {
        if (row[0] === "Per-token elements") {
          return ["Per-device KV elements per token", row[1] / tp, row[2]];
        }
        return row.slice();
      })
    };
  }

  return {
    calculateView: calculateView,
    draftDetailsForView: draftDetailsForView,
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
    speculativeModelsFor: speculativeModelsFor,
    tpHelpText: tpHelpText,
    validTpSizes: validTpSizes
  };
});
