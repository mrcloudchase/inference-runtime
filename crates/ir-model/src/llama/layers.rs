use crate::error::Result;
use crate::gguf::reader::GgufFile;
use super::config::LlamaConfig;

/// Weight tensors for a single LLaMA transformer layer.
///
/// All weights are stored as flat f32 vectors in row-major order.
pub struct LlamaLayer {
    /// RMS norm weights for the attention sub-layer, length = n_embd.
    pub attn_norm: Vec<f32>,
    /// Query projection weights, shape [n_heads * head_dim, n_embd].
    pub wq: Vec<f32>,
    /// Key projection weights, shape [n_kv_heads * head_dim, n_embd].
    pub wk: Vec<f32>,
    /// Value projection weights, shape [n_kv_heads * head_dim, n_embd].
    pub wv: Vec<f32>,
    /// Output projection weights, shape [n_embd, n_heads * head_dim].
    pub wo: Vec<f32>,
    /// RMS norm weights for the FFN sub-layer, length = n_embd.
    pub ffn_norm: Vec<f32>,
    /// Gate projection weights (w1), shape [n_ff, n_embd].
    pub ffn_gate: Vec<f32>,
    /// Up projection weights (w3), shape [n_ff, n_embd].
    pub ffn_up: Vec<f32>,
    /// Down projection weights (w2), shape [n_embd, n_ff].
    pub ffn_down: Vec<f32>,
}

/// All weight tensors for a LLaMA model.
pub struct LlamaWeights {
    /// Token embedding matrix, shape [n_vocab, n_embd].
    pub token_embd: Vec<f32>,
    /// Final RMS norm weights, length = n_embd.
    pub output_norm: Vec<f32>,
    /// Output (LM head) projection weights, shape [n_vocab, n_embd].
    pub output: Vec<f32>,
    /// Per-layer weights.
    pub layers: Vec<LlamaLayer>,
}

impl LlamaWeights {
    /// Load all LLaMA weights from a parsed GGUF file.
    ///
    /// GGUF tensor names follow this pattern:
    /// - `token_embd.weight`
    /// - `output_norm.weight`
    /// - `output.weight` (falls back to token_embd if not present, for tied embeddings)
    /// - `blk.{i}.attn_norm.weight`
    /// - `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`
    /// - `blk.{i}.attn_output.weight`
    /// - `blk.{i}.ffn_norm.weight`
    /// - `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`
    pub fn from_gguf(gguf: &GgufFile, config: &LlamaConfig) -> Result<LlamaWeights> {
        let token_embd = gguf.get_tensor_f32("token_embd.weight")?.data_f32().to_vec();
        let output_norm = gguf.get_tensor_f32("output_norm.weight")?.data_f32().to_vec();

        // Output weights may not exist if embeddings are tied.
        let output = match gguf.get_tensor_f32("output.weight") {
            Ok(t) => t.data_f32().to_vec(),
            Err(_) => token_embd.clone(),
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let attn_norm = gguf
                .get_tensor_f32(&format!("blk.{}.attn_norm.weight", i))?
                .data_f32()
                .to_vec();
            let wq = gguf
                .get_tensor_f32(&format!("blk.{}.attn_q.weight", i))?
                .data_f32()
                .to_vec();
            let wk = gguf
                .get_tensor_f32(&format!("blk.{}.attn_k.weight", i))?
                .data_f32()
                .to_vec();
            let wv = gguf
                .get_tensor_f32(&format!("blk.{}.attn_v.weight", i))?
                .data_f32()
                .to_vec();
            let wo = gguf
                .get_tensor_f32(&format!("blk.{}.attn_output.weight", i))?
                .data_f32()
                .to_vec();
            let ffn_norm = gguf
                .get_tensor_f32(&format!("blk.{}.ffn_norm.weight", i))?
                .data_f32()
                .to_vec();
            let ffn_gate = gguf
                .get_tensor_f32(&format!("blk.{}.ffn_gate.weight", i))?
                .data_f32()
                .to_vec();
            let ffn_up = gguf
                .get_tensor_f32(&format!("blk.{}.ffn_up.weight", i))?
                .data_f32()
                .to_vec();
            let ffn_down = gguf
                .get_tensor_f32(&format!("blk.{}.ffn_down.weight", i))?
                .data_f32()
                .to_vec();

            layers.push(LlamaLayer {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(LlamaWeights {
            token_embd,
            output_norm,
            output,
            layers,
        })
    }
}
