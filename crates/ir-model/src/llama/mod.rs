pub mod config;
pub mod kv_cache;
pub mod layers;

pub use config::LlamaConfig;
pub use kv_cache::KvCache;
pub use layers::{LlamaLayer, LlamaWeights};

use ir_tensor::ComputeBackend;

use crate::architecture::ModelArchitecture;
use crate::error::{ModelError, Result};
use crate::gguf::reader::GgufFile;

/// A LLaMA transformer model loaded from a GGUF file.
///
/// Holds the configuration, dequantized weights (as f32), and a KV cache
/// for autoregressive generation.
pub struct LlamaModel {
    /// Model hyperparameters.
    pub config: LlamaConfig,
    /// All weight tensors (dequantized to f32).
    pub weights: LlamaWeights,
    /// Key-value cache for attention.
    pub cache: KvCache,
}

impl LlamaModel {
    /// Load a LLaMA model from a parsed GGUF file.
    ///
    /// Parses the configuration from metadata, loads and dequantizes all
    /// weight tensors, and initializes an empty KV cache.
    pub fn from_gguf(gguf: &GgufFile, _backend: &dyn ComputeBackend) -> Result<LlamaModel> {
        let config = LlamaConfig::from_gguf(&gguf.metadata)?;
        let weights = LlamaWeights::from_gguf(gguf, &config)?;
        let cache = KvCache::new(
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            config.max_seq_len,
        );

        Ok(LlamaModel {
            config,
            weights,
            cache,
        })
    }

    /// Returns a reference to the model configuration.
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
}

impl ModelArchitecture for LlamaModel {
    /// Run the full LLaMA transformer forward pass.
    ///
    /// Processes each input token through embedding lookup, all transformer
    /// layers (attention + FFN with residual connections), final layer norm,
    /// and the output projection to produce logits for the last token.
    ///
    /// Supports Grouped Query Attention (GQA) where n_kv_heads <= n_heads.
    fn forward(
        &mut self,
        tokens: &[u32],
        pos: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let n_layers = cfg.n_layers;
        let heads_per_kv = n_heads / n_kv_heads;

        let n_tokens = tokens.len();
        if n_tokens == 0 {
            return Err(ModelError::Other("no tokens to process".to_string()));
        }

        // We will compute logits for the last token only.
        let mut last_logits = Vec::new();

        for (t_idx, &token_id) in tokens.iter().enumerate() {
            let cur_pos = pos + t_idx;

            // Step 1: Embedding lookup.
            if (token_id as usize) >= cfg.n_vocab {
                return Err(ModelError::Other(format!(
                    "token id {} exceeds vocab size {}",
                    token_id, cfg.n_vocab
                )));
            }
            let embd_offset = token_id as usize * n_embd;
            let mut hidden: Vec<f32> =
                self.weights.token_embd[embd_offset..embd_offset + n_embd].to_vec();

            // Step 2: Process each transformer layer.
            for layer_idx in 0..n_layers {
                let layer = &self.weights.layers[layer_idx];

                // 2a. RMS norm for attention sub-layer.
                let normed = backend
                    .rms_norm(&hidden, &layer.attn_norm, cfg.norm_eps, n_embd)
                    .map_err(|e| ModelError::Other(format!("rms_norm failed: {}", e)))?;

                // 2b. Compute Q, K, V projections.
                //
                // GGUF stores weight matrices in [out_dim, in_dim] row-major layout.
                // For a single token (vector of length n_embd), we compute the
                // matrix-vector product W @ x using matmul(W, x, out_dim, in_dim, 1).
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv_heads * head_dim;

                let q = backend
                    .matmul(&layer.wq, &normed, q_dim, n_embd, 1)
                    .map_err(|e| ModelError::Other(format!("q matmul failed: {}", e)))?;
                let k = backend
                    .matmul(&layer.wk, &normed, kv_dim, n_embd, 1)
                    .map_err(|e| ModelError::Other(format!("k matmul failed: {}", e)))?;
                let v = backend
                    .matmul(&layer.wv, &normed, kv_dim, n_embd, 1)
                    .map_err(|e| ModelError::Other(format!("v matmul failed: {}", e)))?;

                // 2c. Apply RoPE to Q and K.
                let (q_roped, k_roped) = backend
                    .rope(&q, &k, head_dim, cur_pos, n_heads, n_kv_heads)
                    .map_err(|e| ModelError::Other(format!("rope failed: {}", e)))?;

                // 2d. Update KV cache.
                self.cache.update(layer_idx, &k_roped, &v, cur_pos);

                // Total positions in cache including this one.
                let seq_len = cur_pos + 1;

                // 2e. Compute attention with GQA.
                let cached_k = self.cache.get_k(layer_idx, seq_len);
                let cached_v = self.cache.get_v(layer_idx, seq_len);

                let mut attn_output = vec![0.0f32; q_dim];
                let scale = 1.0 / (head_dim as f32).sqrt();

                for h in 0..n_heads {
                    let kv_h = h / heads_per_kv;

                    // Query vector for this head.
                    let q_start = h * head_dim;
                    let q_head = &q_roped[q_start..q_start + head_dim];

                    // Compute attention scores against all cached keys.
                    let mut scores = Vec::with_capacity(seq_len);
                    for s in 0..seq_len {
                        let k_offset = s * kv_dim + kv_h * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_head[d] * cached_k[k_offset + d];
                        }
                        scores.push(dot * scale);
                    }

                    // Causal masking is implicit: the cache only contains
                    // positions 0..seq_len which are all <= cur_pos.

                    // Softmax over scores (inline for efficiency with single head).
                    let max_score = scores
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    let mut probs = Vec::with_capacity(seq_len);
                    for &s in &scores {
                        let e = (s - max_score).exp();
                        probs.push(e);
                        exp_sum += e;
                    }
                    for p in &mut probs {
                        *p /= exp_sum;
                    }

                    // Weighted sum of cached values.
                    let attn_start = h * head_dim;
                    for (s, &prob) in probs.iter().enumerate().take(seq_len) {
                        let v_offset = s * kv_dim + kv_h * head_dim;
                        for d in 0..head_dim {
                            attn_output[attn_start + d] +=
                                prob * cached_v[v_offset + d];
                        }
                    }
                }

                // 2f. Output projection: wo @ attn_output -> [n_embd].
                let attn_proj = backend
                    .matmul(&layer.wo, &attn_output, n_embd, q_dim, 1)
                    .map_err(|e| ModelError::Other(format!("wo matmul failed: {}", e)))?;

                // 2g. Residual connection.
                hidden = backend
                    .add(&hidden, &attn_proj)
                    .map_err(|e| ModelError::Other(format!("residual add failed: {}", e)))?;

                // 2h. RMS norm for FFN sub-layer.
                let ffn_normed = backend
                    .rms_norm(&hidden, &layer.ffn_norm, cfg.norm_eps, n_embd)
                    .map_err(|e| {
                        ModelError::Other(format!("ffn rms_norm failed: {}", e))
                    })?;

                // 2i. FFN: SwiGLU.
                //   gate = silu(ffn_gate @ normed)  -> [n_ff]
                //   up   = ffn_up @ normed          -> [n_ff]
                //   out  = ffn_down @ (gate * up)   -> [n_embd]
                let gate = backend
                    .matmul(&layer.ffn_gate, &ffn_normed, cfg.n_ff, n_embd, 1)
                    .map_err(|e| ModelError::Other(format!("gate matmul failed: {}", e)))?;
                let up = backend
                    .matmul(&layer.ffn_up, &ffn_normed, cfg.n_ff, n_embd, 1)
                    .map_err(|e| ModelError::Other(format!("up matmul failed: {}", e)))?;
                let gate_activated = backend
                    .silu(&gate)
                    .map_err(|e| ModelError::Other(format!("silu failed: {}", e)))?;
                let gate_up = backend
                    .mul(&gate_activated, &up)
                    .map_err(|e| ModelError::Other(format!("gate*up failed: {}", e)))?;
                let ffn_out = backend
                    .matmul(&layer.ffn_down, &gate_up, n_embd, cfg.n_ff, 1)
                    .map_err(|e| {
                        ModelError::Other(format!("down matmul failed: {}", e))
                    })?;

                // 2j. Residual connection.
                hidden = backend
                    .add(&hidden, &ffn_out)
                    .map_err(|e| {
                        ModelError::Other(format!("ffn residual add failed: {}", e))
                    })?;
            }

            // Step 3: Final RMS norm + LM head (only for last token).
            if t_idx == n_tokens - 1 {
                let final_normed = backend
                    .rms_norm(
                        &hidden,
                        &self.weights.output_norm,
                        cfg.norm_eps,
                        n_embd,
                    )
                    .map_err(|e| {
                        ModelError::Other(format!("output rms_norm failed: {}", e))
                    })?;

                // Step 4: Output projection -> logits [n_vocab].
                last_logits = backend
                    .matmul(
                        &self.weights.output,
                        &final_normed,
                        cfg.n_vocab,
                        n_embd,
                        1,
                    )
                    .map_err(|e| {
                        ModelError::Other(format!("logits matmul failed: {}", e))
                    })?;
            }
        }

        Ok(last_logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.n_vocab
    }

    fn reset_cache(&mut self) {
        self.cache.reset();
    }
}
