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
      + ". Only divisors are shown because num_key_value_heads / TP size must be an integer,"
      + " giving each device a whole number of KV heads.";
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
        precision: normalizePrecision(input.precision, defaultPrecision(model)),
        indexerPrecision: normalizePrecision(
          input.indexerPrecision,
          defaultIndexerPrecision(model)
        ),
        includeDraftKvCache: Boolean(input.includeDraftKvCache),
        includeLinearAttentionState: Boolean(input.includeLinearAttentionState),
        tensorParallel: requestedTp
      },
      precisionConfig(sourceData)
    );

    const deviceGroups = result.cacheGroups.map(function (group) {
      return Object.assign({}, group);
    });
    const perDeviceBytes = result.perDeviceBytes;
    const allDeviceBytes = result.allDeviceBytes;

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

  function groupRatio(group, fallbackTp) {
    if (group && Number.isFinite(group.localHeads) && Number.isFinite(group.heads)) {
      return group.localHeads / group.heads;
    }
    return 1 / fallbackTp;
  }

  function scaledDetail(label, value, help, ratio, nextLabel) {
    return [
      nextLabel || label,
      typeof value === "number" ? value * ratio : value,
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
    const groupsByLabel = Object.fromEntries(view.deviceGroups.map(function (group) {
      return [group.label, group];
    }));
    const kvGroup = view.deviceGroups.find(function (group) { return group.role === "kv"; });
    const indexerGroup = view.deviceGroups.find(function (group) { return group.role === "indexer"; });
    const linearGroup = view.deviceGroups.find(function (group) { return group.role === "linear_state"; });
    const rows = [];

    original.forEach(function (row) {
      const label = row[0];
      const value = row[1];
      const help = row[2];

      if ((formula === "standard_gqa" || formula === "mla") && label === "Per-token elements") {
        rows.push(scaledDetail(
          label,
          value,
          help,
          groupRatio(kvGroup, tp),
          "Per-device KV elements per token"
        ));
        return;
      }

      if (formula === "dsa_mla") {
        if (label === "KV elements per token") {
          rows.push(scaledDetail(label, value, help, groupRatio(kvGroup, tp), "Per-device KV elements per token"));
          return;
        }
        if (label === "Indexer elements per token") {
          rows.push(scaledDetail(label, value, help, groupRatio(indexerGroup, tp), "Per-device Indexer elements per token"));
          return;
        }
        if (label === "Per-token elements") {
          const kv = Number(values["KV elements per token"]) || 0;
          const indexer = Number(values["Indexer elements per token"]) || 0;
          rows.push([
            "Per-device total elements per token",
            kv * groupRatio(kvGroup, tp) + indexer * groupRatio(indexerGroup, tp),
            help
          ]);
          return;
        }
      }

      if (formula === "qwen_linear_full_hybrid") {
        if (label === "Linear conv elements") {
          rows.push(scaledDetail(label, value, help, groupRatio(linearGroup, tp), "Per-device linear conv elements"));
          return;
        }
        if (label === "Linear recurrent elements") {
          rows.push(scaledDetail(label, value, help, groupRatio(linearGroup, tp), "Per-device linear recurrent elements"));
          return;
        }
        if (label === "Per-token elements") {
          rows.push(scaledDetail(
            label,
            value,
            help,
            groupRatio(kvGroup, tp),
            "Per-device KV elements per token"
          ));
          return;
        }
      }

      if (formula === "mixed_full_sliding_gqa") {
        if (label === "Full-attention elements") {
          rows.push(scaledDetail(label, value, help, groupRatio(groupsByLabel["Full-attention KV cache"], tp), "Per-device full-attention elements"));
          return;
        }
        if (label === "Sliding-window elements") {
          rows.push(scaledDetail(label, value, help, groupRatio(groupsByLabel["Sliding-window KV cache"], tp), "Per-device sliding-window elements"));
          return;
        }
      }

      if (formula === "minimax_msa") {
        if (label === "KV elements per token") {
          rows.push(scaledDetail(label, value, help, groupRatio(kvGroup, tp), "Per-device KV elements per token"));
          return;
        }
        if (label === "Indexer elements per token") {
          rows.push(scaledDetail(label, value, help, groupRatio(indexerGroup, tp), "Per-device Indexer elements per token"));
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
          rows.push(scaledDetail(label, value, help, groupRatio(kvGroup, tp), "Per-device " + label.toLowerCase()));
          return;
        }
        if (label === "Indexer elements") {
          rows.push(scaledDetail(label, value, help, groupRatio(indexerGroup, tp), "Per-device Indexer elements"));
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
    const formula = view.model.formula;
    const rows = (view.elementPlan.formulaRows || []).map(function (row) {
      return {
        name: row.name,
        expression: row.expression,
        description: row.description
      };
    });
    function withoutSequences(expression) {
      return expression
        .replace(/tokens x sequences x /g, "tokens x ")
        .replace(/min\(tokens, sliding_window\) x sequences x /g, "min(tokens, sliding_window) x ")
        .replace(/^sequences x /, "");
    }
    function localHeads(expression, field) {
      return expression.replace(
        new RegExp("\\b" + field + "\\b", "g"),
        "max(" + field + " / TP size, 1)"
      );
    }
    function rowWith(row, name, expression, description) {
      return {
        name: name || row.name,
        expression: expression,
        description: description || row.description
      };
    }

    let terms;
    const perDeviceRows = [];

    rows.forEach(function (row) {
      let expression = withoutSequences(row.expression);

      if (row.name === "total_bytes") {
        return;
      }

      if (formula === "standard_gqa" && row.name !== "active_layers") {
        expression = localHeads(expression, "num_key_value_heads");
      }
      if (formula === "dsa_mla" && row.name === "indexer_bytes") {
        expression = localHeads(expression, "index_num_key_heads");
      }
      if (formula === "qwen_linear_full_hybrid") {
        expression = localHeads(expression, "num_key_value_heads");
        expression = localHeads(expression, "linear_num_key_heads");
        expression = localHeads(expression, "linear_num_value_heads");
      }
      if (formula === "mixed_full_sliding_gqa") {
        expression = localHeads(expression, "full_kv_heads");
        expression = localHeads(expression, "sliding_kv_heads");
      }
      if (formula === "minimax_msa" && row.name === "kv_bytes") {
        expression = localHeads(expression, "num_key_value_heads");
      }
      if ((formula === "minimax_msa" || formula === "deepseek_v4_hybrid") && row.name === "indexer_bytes") {
        expression = localHeads(expression, "index_num_key_heads");
      }

      perDeviceRows.push(rowWith(row, null, expression));
    });

    if (formula === "standard_gqa") {
      const total = rows.find(function (row) { return row.name === "total_bytes"; });
      let expression = withoutSequences(total.expression);
      expression = localHeads(expression, "num_key_value_heads");
      perDeviceRows.push(rowWith(total, "kv_bytes", expression, "Per-sequence KV cache using the local KV-head count."));
      terms = "kv_bytes";
    } else if (formula === "mla") {
      const total = rows.find(function (row) { return row.name === "total_bytes"; });
      perDeviceRows.push(rowWith(total, "kv_bytes", withoutSequences(total.expression), "Per-sequence latent KV cache; its single head is present on every TP device."));
      terms = "kv_bytes";
    } else if (formula === "dsa_mla") {
      terms = "kv_bytes + indexer_bytes";
    } else if (formula === "qwen_linear_full_hybrid") {
      terms = perDeviceRows.some(function (row) { return row.name === "linear_conv_state_bytes"; })
        ? "full_kv_bytes + linear_conv_state_bytes + linear_recurrent_state_bytes"
        : "full_kv_bytes";
    } else if (formula === "mixed_full_sliding_gqa") {
      terms = "full_kv_bytes + sliding_kv_bytes";
    } else {
      terms = "kv_bytes + indexer_bytes";
    }

    const perDevicePayload = terms.includes(" + ") ? "(" + terms + ")" : terms;
    perDeviceRows.push({
      name: "per_device_bytes",
      expression: "sequences × " + perDevicePayload,
      description: "Per-device cache for all concurrent sequences after applying each cache group's local head count."
    });
    perDeviceRows.push({
      name: "all_device_bytes",
      expression: "TP size × per_device_bytes",
      description: "Physical cache across all tensor-parallel devices."
    });
    return perDeviceRows;
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
        const tpNote = " Each cache group uses max(heads / TP size, 1) local heads per device.";
        get("cache-note").textContent = view.elementPlan.note + tpNote;
        renderBreakdown(get("breakdown"), detailsForView(view), doc);

        get("source").textContent = model.source_url;
        get("source-link").href = model.source_url;
      } catch (error) {
        get("cache-note").textContent = error.message;
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
    formulaRowsForView: formulaRowsForView,
    modelsForFamily: modelsForFamily,
    mount: mount,
    tpHelpText: tpHelpText,
    validTpSizes: validTpSizes
  };
});
