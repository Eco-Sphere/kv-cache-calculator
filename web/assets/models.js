/*
 * Model data adapted from kvcache-ai/kvcache-blog at commit
 * d455485bd1c5265951849f011a60564da1b23cfe (MIT License).
 * See ../../THIRD_PARTY_NOTICES.md.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.KV_MODEL_DATA = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  return {
  "indexer_precision_options": [
    {
      "bytes_per_element": 2,
      "id": "bf16_fp16",
      "label": "BF16 / FP16"
    },
    {
      "bytes_per_element": 1,
      "id": "fp8_int8",
      "label": "FP8 / INT8"
    },
    {
      "bytes_per_element": 0.5,
      "id": "fp4_int4",
      "label": "FP4 / INT4"
    }
  ],
  "metadata": {
    "note": "Curated from official Hugging Face model config/source files and serving-engine references. Values describe KV cache capacity planning, not model weights or activation memory.",
    "retrieved_at": "2026-05-23",
    "serving_references": {
      "deepseek_v4_vllm_indexer_fp4": "https://vllm.ai/blog/deepseek-v4"
    }
  },
  "models": [
    {
      "default_tokens": 1024,
      "family": "DeepSeek",
      "fields": {
        "compress_ratios": [
          128,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          0
        ],
        "head_dim": 512,
        "index_head_dim": 128,
        "index_topk": 1024,
        "num_hidden_layers": 61,
        "num_key_value_heads": 1,
        "sliding_window": 128
      },
      "formula": "deepseek_v4_hybrid",
      "id": "deepseek-v4-pro",
      "label": "DeepSeek V4 Pro",
      "max_position_embeddings": 1048576,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "DeepSeek",
      "fields": {
        "compress_ratios": [
          0,
          0,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          128,
          4,
          0
        ],
        "head_dim": 512,
        "index_head_dim": 128,
        "index_topk": 512,
        "num_hidden_layers": 43,
        "num_key_value_heads": 1,
        "sliding_window": 128
      },
      "formula": "deepseek_v4_hybrid",
      "id": "deepseek-v4-flash",
      "label": "DeepSeek V4 Flash",
      "max_position_embeddings": 1048576,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "DeepSeek",
      "fields": {
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 2048,
        "kv_lora_rank": 512,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "num_nextn_predict_layers": 1,
        "qk_head_dim": 192,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
      },
      "formula": "dsa_mla",
      "id": "deepseek-v3.2",
      "label": "DeepSeek V3.2",
      "max_position_embeddings": 163840,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "DeepSeek",
      "fields": {
        "kv_lora_rank": 512,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "num_nextn_predict_layers": 1,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
      },
      "formula": "mla",
      "id": "deepseek-v3",
      "label": "DeepSeek V3",
      "max_position_embeddings": 163840,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "DeepSeek",
      "fields": {
        "kv_lora_rank": 512,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "num_nextn_predict_layers": 1,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
      },
      "formula": "mla",
      "id": "deepseek-r1",
      "label": "DeepSeek R1",
      "max_position_embeddings": 163840,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "GLM",
      "fields": {
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "kv_lora_rank": 512,
        "num_hidden_layers": 78,
        "num_key_value_heads": 64,
        "num_nextn_predict_layers": 1,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256
      },
      "formula": "dsa_mla",
      "id": "glm-5",
      "label": "GLM-5",
      "max_position_embeddings": 202752,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/zai-org/GLM-5/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "GLM",
      "fields": {
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "kv_lora_rank": 512,
        "num_hidden_layers": 78,
        "num_key_value_heads": 64,
        "num_nextn_predict_layers": 1,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256
      },
      "formula": "dsa_mla",
      "id": "glm-5.1",
      "label": "GLM-5.1",
      "max_position_embeddings": 202752,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/zai-org/GLM-5.1/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "GLM",
      "fields": {
        "draft_indexer_layers": 1,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_share_for_mtp_iteration": true,
        "index_skip_topk_offset": 3,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "indexer_full_layers": 21,
        "indexer_shared_layers": 57,
        "kv_lora_rank": 512,
        "num_hidden_layers": 78,
        "num_key_value_heads": 64,
        "num_nextn_predict_layers": 1,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256
      },
      "formula": "dsa_mla",
      "id": "glm-5.2",
      "label": "GLM-5.2",
      "max_position_embeddings": 1048576,
      "source_retrieved_at": "2026-06-17",
      "source_url": "https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Kimi",
      "fields": {
        "kv_lora_rank": 512,
        "num_hidden_layers": 61,
        "num_key_value_heads": 64,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
      },
      "formula": "mla",
      "id": "kimi-k2.5",
      "label": "Kimi K2.5",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/moonshotai/Kimi-K2.5/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Kimi",
      "fields": {
        "kv_lora_rank": 512,
        "num_hidden_layers": 61,
        "num_key_value_heads": 64,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
      },
      "formula": "mla",
      "id": "kimi-k2.6",
      "label": "Kimi K2.6",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.6",
      "fields": {
        "full_attention_layers": 16,
        "head_dim": 256,
        "linear_attention_layers": 48,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.6-27b",
      "label": "Qwen3.6-27B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.6",
      "fields": {
        "full_attention_layers": 10,
        "head_dim": 256,
        "linear_attention_layers": 30,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.6-35b-a3b",
      "label": "Qwen3.6-35B-A3B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 15,
        "head_dim": 256,
        "linear_attention_layers": 45,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 32,
        "num_hidden_layers": 60,
        "num_key_value_heads": 2
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-397b-a17b",
      "label": "Qwen3.5-397B-A17B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 12,
        "head_dim": 256,
        "linear_attention_layers": 36,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 32,
        "num_hidden_layers": 48,
        "num_key_value_heads": 2
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-122b-a10b",
      "label": "Qwen3.5-122B-A10B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-122B-A10B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 10,
        "head_dim": 256,
        "linear_attention_layers": 30,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-35b-a3b",
      "label": "Qwen3.5-35B-A3B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 16,
        "head_dim": 256,
        "linear_attention_layers": 48,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-27b",
      "label": "Qwen3.5-27B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 8,
        "head_dim": 256,
        "linear_attention_layers": 24,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-9b",
      "label": "Qwen3.5-9B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 8,
        "head_dim": 256,
        "linear_attention_layers": 24,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-4b",
      "label": "Qwen3.5-4B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 6,
        "head_dim": 256,
        "linear_attention_layers": 18,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-2b",
      "label": "Qwen3.5-2B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3.5",
      "fields": {
        "full_attention_layers": 6,
        "head_dim": 256,
        "linear_attention_layers": 18,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2
      },
      "formula": "qwen_linear_full_hybrid",
      "id": "qwen3.5-0.8b",
      "label": "Qwen3.5-0.8B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 64,
        "num_hidden_layers": 94,
        "num_key_value_heads": 4
      },
      "formula": "standard_gqa",
      "id": "qwen3-235b-a22b",
      "label": "Qwen3-235B-A22B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 64,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "qwen3-32b",
      "label": "Qwen3-32B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4
      },
      "formula": "standard_gqa",
      "id": "qwen3-30b-a3b",
      "label": "Qwen3-30B-A3B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "qwen3-14b",
      "label": "Qwen3-14B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "qwen3-8b",
      "label": "Qwen3-8B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "qwen3-4b",
      "label": "Qwen3-4B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "qwen3-1.7b",
      "label": "Qwen3-1.7B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen3",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "qwen3-0.6b",
      "label": "Qwen3-0.6B",
      "max_position_embeddings": 40960,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen2.5",
      "fields": {
        "head_dim": 128,
        "max_window_layers": 70,
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "sliding_window": 131072,
        "use_sliding_window": false
      },
      "formula": "standard_gqa",
      "id": "qwen2.5-72b",
      "label": "Qwen2.5-72B",
      "max_position_embeddings": 32768,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen2.5",
      "fields": {
        "head_dim": 128,
        "max_window_layers": 70,
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "sliding_window": 131072,
        "use_sliding_window": false
      },
      "formula": "standard_gqa",
      "id": "qwen2.5-32b",
      "label": "Qwen2.5-32B",
      "max_position_embeddings": 32768,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen2.5",
      "fields": {
        "head_dim": 128,
        "max_window_layers": 70,
        "num_attention_heads": 40,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "sliding_window": 131072,
        "use_sliding_window": false
      },
      "formula": "standard_gqa",
      "id": "qwen2.5-14b",
      "label": "Qwen2.5-14B",
      "max_position_embeddings": 32768,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen2.5",
      "fields": {
        "head_dim": 128,
        "max_window_layers": 28,
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "sliding_window": 131072,
        "use_sliding_window": false
      },
      "formula": "standard_gqa",
      "id": "qwen2.5-7b",
      "label": "Qwen2.5-7B",
      "max_position_embeddings": 32768,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Qwen2.5",
      "fields": {
        "head_dim": 128,
        "max_window_layers": 64,
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "sliding_window": 131072,
        "use_sliding_window": false
      },
      "formula": "standard_gqa",
      "id": "qwen2.5-coder-32b",
      "label": "Qwen2.5-Coder-32B",
      "max_position_embeddings": 32768,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Llama",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "llama-3.1-8b",
      "label": "Llama 3.1 8B",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Llama",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "llama-3.1-70b",
      "label": "Llama 3.1 70B",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Llama",
      "fields": {
        "head_dim": 128,
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "llama-3.3-70b",
      "label": "Llama 3.3 70B",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-22",
      "source_url": "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Gemma",
      "fields": {
        "full_attention_layers": 3,
        "global_head_dim": 512,
        "head_dim": 256,
        "num_attention_heads": 8,
        "num_hidden_layers": 35,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 20,
        "sliding_attention_layers": 12,
        "sliding_window": 512,
        "stored_layers": 15
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "gemma-4-e2b",
      "label": "Gemma 4 E2B",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/google/gemma-4-E2B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Gemma",
      "fields": {
        "full_attention_layers": 4,
        "global_head_dim": 512,
        "head_dim": 256,
        "num_attention_heads": 8,
        "num_hidden_layers": 42,
        "num_key_value_heads": 2,
        "num_kv_shared_layers": 18,
        "sliding_attention_layers": 20,
        "sliding_window": 512,
        "stored_layers": 24
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "gemma-4-e4b",
      "label": "Gemma 4 E4B",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/google/gemma-4-E4B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Gemma",
      "fields": {
        "full_attention_layers": 5,
        "global_head_dim": 512,
        "head_dim": 256,
        "num_attention_heads": 16,
        "num_global_key_value_heads": 2,
        "num_hidden_layers": 30,
        "num_key_value_heads": 8,
        "num_kv_shared_layers": 0,
        "sliding_attention_layers": 25,
        "sliding_window": 1024,
        "stored_layers": 30
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "gemma-4-26b-a4b",
      "label": "Gemma 4 26B-A4B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/google/gemma-4-26B-A4B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Gemma",
      "fields": {
        "full_attention_layers": 10,
        "global_head_dim": 512,
        "head_dim": 256,
        "num_attention_heads": 32,
        "num_global_key_value_heads": 4,
        "num_hidden_layers": 60,
        "num_key_value_heads": 16,
        "num_kv_shared_layers": 0,
        "sliding_attention_layers": 50,
        "sliding_window": 1024,
        "stored_layers": 60
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "gemma-4-31b",
      "label": "Gemma 4 31B",
      "max_position_embeddings": 262144,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/google/gemma-4-31B/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Cohere",
      "fields": {
        "config_max_position_embeddings": 8192,
        "head_dim": 128,
        "model_max_length": 131072,
        "num_attention_heads": 64,
        "num_hidden_layers": 40,
        "num_key_value_heads": 64
      },
      "formula": "standard_gqa",
      "id": "cohere-command-r-v01",
      "label": "Cohere Command R v01",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/CohereLabs/c4ai-command-r-v01/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Cohere",
      "fields": {
        "config_max_position_embeddings": 8192,
        "head_dim": 128,
        "model_max_length": 131072,
        "num_attention_heads": 96,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8
      },
      "formula": "standard_gqa",
      "id": "cohere-command-r-plus",
      "label": "Cohere Command R+",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/CohereLabs/c4ai-command-r-plus/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Cohere",
      "fields": {
        "full_attention_layers": 8,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "sliding_attention_layers": 24,
        "sliding_window": 4096,
        "sliding_window_pattern": 4
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "cohere-command-r7b-12-2024",
      "label": "Cohere Command R7B 12-2024",
      "max_position_embeddings": 132096,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Cohere",
      "fields": {
        "full_attention_layers": 16,
        "head_dim": 128,
        "num_attention_heads": 96,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "sliding_attention_layers": 48,
        "sliding_window": 4096,
        "sliding_window_pattern": 4
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "cohere-command-a-03-2025",
      "label": "Cohere Command A 03-2025",
      "max_position_embeddings": 131072,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/CohereLabs/c4ai-command-a-03-2025/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "Cohere",
      "fields": {
        "full_attention_layers": 8,
        "head_dim": 128,
        "num_attention_heads": 128,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "sliding_attention_layers": 24,
        "sliding_window": 4096
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "cohere-command-a-plus-05-2026",
      "label": "Cohere Command A Plus 05-2026",
      "max_position_embeddings": 200000,
      "source_retrieved_at": "2026-05-23",
      "source_url": "https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "MiMo",
      "fields": {
        "full_attention_layers": 9,
        "head_dim": 192,
        "num_attention_heads": 64,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "sliding_attention_layers": 39,
        "sliding_window": 128,
        "sliding_window_size": 128,
        "swa_head_dim": 192,
        "swa_num_attention_heads": 64,
        "swa_num_key_value_heads": 8,
        "swa_v_head_dim": 128,
        "v_head_dim": 128
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "mimo-v2.5",
      "label": "MiMo-V2.5",
      "max_position_embeddings": 1048576,
      "source_retrieved_at": "2026-05-27",
      "source_url": "https://huggingface.co/XiaomiMiMo/MiMo-V2.5/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "MiMo",
      "fields": {
        "full_attention_layers": 10,
        "head_dim": 192,
        "num_attention_heads": 128,
        "num_hidden_layers": 70,
        "num_key_value_heads": 8,
        "sliding_attention_layers": 60,
        "sliding_window": 128,
        "sliding_window_size": 128,
        "swa_head_dim": 192,
        "swa_num_attention_heads": 128,
        "swa_num_key_value_heads": 8,
        "swa_v_head_dim": 128,
        "v_head_dim": 128
      },
      "formula": "mixed_full_sliding_gqa",
      "id": "mimo-v2.5-pro",
      "label": "MiMo-V2.5-Pro",
      "max_position_embeddings": 1048576,
      "source_retrieved_at": "2026-05-27",
      "source_url": "https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "MiniMax",
      "fields": {
        "head_dim": 128,
        "mtp_transformer_layers": 1,
        "num_attention_heads": 48,
        "num_hidden_layers": 62,
        "num_key_value_heads": 8,
        "num_mtp_modules": 3,
        "rotary_dim": 64,
        "use_mtp": true
      },
      "formula": "standard_gqa",
      "id": "minimax-m2",
      "label": "MiniMax M2",
      "max_position_embeddings": 196608,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "MiniMax",
      "fields": {
        "head_dim": 128,
        "mtp_transformer_layers": 1,
        "num_attention_heads": 48,
        "num_hidden_layers": 62,
        "num_key_value_heads": 8,
        "num_mtp_modules": 3,
        "rotary_dim": 64,
        "use_mtp": true
      },
      "formula": "standard_gqa",
      "id": "minimax-m2.1",
      "label": "MiniMax M2.1",
      "max_position_embeddings": 196608,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/MiniMaxAI/MiniMax-M2.1/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "MiniMax",
      "fields": {
        "head_dim": 128,
        "mtp_transformer_layers": 1,
        "num_attention_heads": 48,
        "num_hidden_layers": 62,
        "num_key_value_heads": 8,
        "num_mtp_modules": 3,
        "rotary_dim": 64,
        "use_mtp": true
      },
      "formula": "standard_gqa",
      "id": "minimax-m2.5",
      "label": "MiniMax M2.5",
      "max_position_embeddings": 196608,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/MiniMaxAI/MiniMax-M2.5/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "MiniMax",
      "fields": {
        "head_dim": 128,
        "mtp_transformer_layers": 1,
        "num_attention_heads": 48,
        "num_hidden_layers": 62,
        "num_key_value_heads": 8,
        "num_mtp_modules": 3,
        "rotary_dim": 64,
        "use_mtp": true
      },
      "formula": "standard_gqa",
      "id": "minimax-m2.7",
      "label": "MiniMax M2.7",
      "max_position_embeddings": 204800,
      "source_retrieved_at": "2026-05-20",
      "source_url": "https://huggingface.co/MiniMaxAI/MiniMax-M2.7/raw/main/config.json"
    },
    {
      "default_tokens": 1024,
      "family": "MiniMax",
      "fields": {
        "disable_draft_kv_cache": true,
        "full_attention_layers": 3,
        "head_dim": 128,
        "index_block_size": 128,
        "index_head_dim": 128,
        "index_local_blocks": 1,
        "index_n_heads": 4,
        "index_topk_blocks": 16,
        "indexer_fixed_precision_id": "bf16_fp16",
        "num_attention_heads": 64,
        "num_hidden_layers": 60,
        "num_key_value_heads": 4,
        "num_mtp_modules": 7,
        "num_nextn_predict_layers": 1,
        "sparse_attention": true,
        "sparse_attention_layers": 57
      },
      "formula": "minimax_msa",
      "id": "minimax-m3",
      "label": "MiniMax M3",
      "max_position_embeddings": 1048576,
      "source_retrieved_at": "2026-06-23",
      "source_url": "https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json"
    }
  ],
  "precision_options": [
    {
      "bytes_per_element": 2,
      "id": "bf16_fp16",
      "label": "BF16 / FP16"
    },
    {
      "bytes_per_element": 1,
      "id": "fp8_int8",
      "label": "FP8 / INT8"
    },
    {
      "bytes_per_element": 0.5,
      "id": "fp4_int4",
      "label": "FP4 / INT4"
    }
  ]
};
});
