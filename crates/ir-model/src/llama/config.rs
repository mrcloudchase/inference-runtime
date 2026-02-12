use crate::error::Result;
use crate::gguf::metadata::GgufMetadata;

/// Configuration for a LLaMA model, parsed from GGUF metadata.
pub struct LlamaConfig {
    /// Vocabulary size (number of token embeddings).
    pub n_vocab: usize,
    /// Embedding dimension / hidden size.
    pub n_embd: usize,
    /// Number of attention heads for queries.
    pub n_heads: usize,
    /// Number of attention heads for keys/values (GQA).
    pub n_kv_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Feed-forward intermediate dimension.
    pub n_ff: usize,
    /// RMS normalization epsilon.
    pub norm_eps: f32,
    /// Maximum sequence length / context window size.
    pub max_seq_len: usize,
    /// RoPE frequency base (theta).
    pub rope_theta: f32,
    /// Dimension of each attention head (n_embd / n_heads).
    pub head_dim: usize,
}

impl LlamaConfig {
    /// Parse a LLaMA configuration from GGUF metadata.
    ///
    /// Reads the following keys:
    /// - `llama.embedding_length` -> n_embd
    /// - `llama.attention.head_count` -> n_heads
    /// - `llama.attention.head_count_kv` -> n_kv_heads
    /// - `llama.block_count` -> n_layers
    /// - `llama.feed_forward_length` -> n_ff
    /// - `llama.attention.layer_norm_rms_epsilon` -> norm_eps
    /// - `llama.context_length` -> max_seq_len
    /// - `llama.rope.freq_base` -> rope_theta (default 10000.0)
    /// - vocab size inferred from `tokenizer.ggml.tokens` array length
    pub fn from_gguf(metadata: &GgufMetadata) -> Result<LlamaConfig> {
        let n_embd = metadata.get_u32("llama.embedding_length")? as usize;
        let n_heads = metadata.get_u32("llama.attention.head_count")? as usize;
        let n_kv_heads = metadata.get_u32("llama.attention.head_count_kv")? as usize;
        let n_layers = metadata.get_u32("llama.block_count")? as usize;
        let n_ff = metadata.get_u32("llama.feed_forward_length")? as usize;
        let norm_eps = metadata.get_f32("llama.attention.layer_norm_rms_epsilon")?;
        let max_seq_len = metadata.get_u32("llama.context_length")? as usize;

        let rope_theta = metadata.get_f32("llama.rope.freq_base").unwrap_or(10000.0);

        // Infer vocab size from tokenizer token array.
        let tokens = metadata.get_string_array("tokenizer.ggml.tokens")?;
        let n_vocab = tokens.len();

        let head_dim = n_embd / n_heads;

        Ok(LlamaConfig {
            n_vocab,
            n_embd,
            n_heads,
            n_kv_heads,
            n_layers,
            n_ff,
            norm_eps,
            max_seq_len,
            rope_theta,
            head_dim,
        })
    }
}
