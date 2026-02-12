use ir_tensor::ComputeBackend;

/// Trait for model architectures that can perform autoregressive inference.
///
/// Implementations hold model weights and KV caches, and can process tokens
/// through the full transformer forward pass to produce next-token logits.
pub trait ModelArchitecture: Send + Sync {
    /// Run the forward pass for a batch of input tokens starting at a given
    /// position in the sequence.
    ///
    /// Returns a vector of logits over the vocabulary for the last token.
    ///
    /// - `tokens`: the input token IDs to process.
    /// - `pos`: the starting position in the sequence (for KV cache and RoPE).
    /// - `backend`: the compute backend to use for tensor operations.
    fn forward(
        &mut self,
        tokens: &[u32],
        pos: usize,
        backend: &dyn ComputeBackend,
    ) -> crate::Result<Vec<f32>>;

    /// Returns the vocabulary size (number of output logits).
    fn vocab_size(&self) -> usize;

    /// Reset all KV caches, clearing any stored context.
    fn reset_cache(&mut self);
}
