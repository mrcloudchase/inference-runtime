use std::fmt::Debug;

use crate::error::Result;

/// Trait for pluggable compute backends (CPU, Metal, CUDA, etc.).
///
/// All operations work on f32 slices for Phase 1. Data is passed in as slices
/// and returned as owned vectors. The backend is responsible for performing
/// the computation and returning the result.
pub trait ComputeBackend: Send + Sync + Debug {
    /// Returns the name of this backend (e.g., "cpu", "metal").
    fn name(&self) -> &str;

    /// Matrix multiplication: C = A @ B.
    ///
    /// - `a`: row-major data of shape [m, k]
    /// - `b`: row-major data of shape [k, n]
    /// - Returns: row-major data of shape [m, n]
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>>;

    /// Element-wise addition: result[i] = a[i] + b[i].
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// Element-wise multiplication: result[i] = a[i] * b[i].
    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    /// Scalar multiplication: result[i] = a[i] * s.
    fn scale(&self, a: &[f32], s: f32) -> Result<Vec<f32>>;

    /// RMS normalization.
    ///
    /// For each row of `hidden_size` elements in `x`:
    ///   rms = sqrt(mean(x^2) + eps)
    ///   result[i] = x[i] * weight[i] / rms
    ///
    /// - `x`: input data, length must be a multiple of `hidden_size`
    /// - `weight`: per-element scale weights, length == `hidden_size`
    /// - `eps`: small constant for numerical stability
    /// - `hidden_size`: size of each row to normalize
    fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        hidden_size: usize,
    ) -> Result<Vec<f32>>;

    /// Softmax over chunks of `n_vocab` elements.
    ///
    /// For each chunk: result[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
    fn softmax(&self, x: &[f32], n_vocab: usize) -> Result<Vec<f32>>;

    /// Rotary Position Embedding (RoPE).
    ///
    /// Applies rotary embeddings to query and key tensors.
    ///
    /// - `q`: query data, shape [n_heads_q, head_dim]
    /// - `k`: key data, shape [n_heads_k, head_dim]
    /// - `head_dim`: dimension of each attention head
    /// - `pos`: token position for computing rotation angles
    /// - `n_heads_q`: number of query heads
    /// - `n_heads_k`: number of key heads
    ///
    /// Returns (rotated_q, rotated_k).
    fn rope(
        &self,
        q: &[f32],
        k: &[f32],
        head_dim: usize,
        pos: usize,
        n_heads_q: usize,
        n_heads_k: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)>;

    /// SiLU activation: result[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])).
    fn silu(&self, x: &[f32]) -> Result<Vec<f32>>;
}
