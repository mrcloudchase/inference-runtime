pub mod matmul;
pub mod unary;

use crate::backend::ComputeBackend;
use crate::error::{Result, TensorError};

/// Pure-Rust CPU compute backend.
///
/// Implements all operations with straightforward loops optimized for
/// correctness rather than peak performance. Intended as a reference
/// implementation and fallback.
#[derive(Debug, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        if a.len() != m * k {
            return Err(TensorError::Other(format!(
                "matmul: a.len()={} but expected m*k={}",
                a.len(),
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(TensorError::Other(format!(
                "matmul: b.len()={} but expected k*n={}",
                b.len(),
                k * n
            )));
        }

        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![a.len()],
                got: vec![b.len()],
            });
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![a.len()],
                got: vec![b.len()],
            });
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
    }

    fn scale(&self, a: &[f32], s: f32) -> Result<Vec<f32>> {
        Ok(a.iter().map(|x| x * s).collect())
    }

    fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        if weight.len() != hidden_size {
            return Err(TensorError::Other(format!(
                "rms_norm: weight.len()={} but hidden_size={}",
                weight.len(),
                hidden_size
            )));
        }
        if x.len() % hidden_size != 0 {
            return Err(TensorError::Other(format!(
                "rms_norm: x.len()={} is not a multiple of hidden_size={}",
                x.len(),
                hidden_size
            )));
        }

        let n_rows = x.len() / hidden_size;
        let mut result = vec![0.0f32; x.len()];

        for row in 0..n_rows {
            let offset = row * hidden_size;
            let row_data = &x[offset..offset + hidden_size];

            // Compute mean of squares
            let mean_sq: f32 =
                row_data.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32;
            let rms = (mean_sq + eps).sqrt();

            // Normalize and scale by weight
            for i in 0..hidden_size {
                result[offset + i] = row_data[i] * weight[i] / rms;
            }
        }

        Ok(result)
    }

    fn softmax(&self, x: &[f32], n_vocab: usize) -> Result<Vec<f32>> {
        if n_vocab == 0 {
            return Err(TensorError::Other(
                "softmax: n_vocab must be > 0".to_string(),
            ));
        }
        if x.len() % n_vocab != 0 {
            return Err(TensorError::Other(format!(
                "softmax: x.len()={} is not a multiple of n_vocab={}",
                x.len(),
                n_vocab
            )));
        }

        let n_chunks = x.len() / n_vocab;
        let mut result = vec![0.0f32; x.len()];

        for chunk in 0..n_chunks {
            let offset = chunk * n_vocab;
            let chunk_data = &x[offset..offset + n_vocab];

            // Find max for numerical stability
            let max_val = chunk_data
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for i in 0..n_vocab {
                let e = (chunk_data[i] - max_val).exp();
                result[offset + i] = e;
                sum += e;
            }

            // Normalize
            for i in 0..n_vocab {
                result[offset + i] /= sum;
            }
        }

        Ok(result)
    }

    fn rope(
        &self,
        q: &[f32],
        k: &[f32],
        head_dim: usize,
        pos: usize,
        n_heads_q: usize,
        n_heads_k: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if q.len() != n_heads_q * head_dim {
            return Err(TensorError::Other(format!(
                "rope: q.len()={} but expected n_heads_q*head_dim={}",
                q.len(),
                n_heads_q * head_dim
            )));
        }
        if k.len() != n_heads_k * head_dim {
            return Err(TensorError::Other(format!(
                "rope: k.len()={} but expected n_heads_k*head_dim={}",
                k.len(),
                n_heads_k * head_dim
            )));
        }

        let mut q_out = q.to_vec();
        let mut k_out = k.to_vec();

        // Apply RoPE to query heads
        for h in 0..n_heads_q {
            let offset = h * head_dim;
            for i in 0..head_dim / 2 {
                let theta =
                    pos as f32 * (1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32));
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let x0 = q[offset + 2 * i];
                let x1 = q[offset + 2 * i + 1];
                q_out[offset + 2 * i] = x0 * cos_theta - x1 * sin_theta;
                q_out[offset + 2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }

        // Apply RoPE to key heads
        for h in 0..n_heads_k {
            let offset = h * head_dim;
            for i in 0..head_dim / 2 {
                let theta =
                    pos as f32 * (1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32));
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let x0 = k[offset + 2 * i];
                let x1 = k[offset + 2 * i + 1];
                k_out[offset + 2 * i] = x0 * cos_theta - x1 * sin_theta;
                k_out[offset + 2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }

        Ok((q_out, k_out))
    }

    fn silu(&self, x: &[f32]) -> Result<Vec<f32>> {
        Ok(x.iter().map(|&v| v / (1.0 + (-v).exp())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> CpuBackend {
        CpuBackend::new()
    }

    #[test]
    fn test_matmul_identity() {
        let b = backend();
        // 2x2 identity @ [1,2;3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let c = b.matmul(&a, &x, 2, 2, 2).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_basic() {
        let b = backend();
        // [1,2] @ [3;4] = [11]
        // [1,2;3,4] @ [5,6;7,8] = [19,22;43,50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0, 7.0, 8.0];
        let c = b.matmul(&a, &x, 2, 2, 2).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_add() {
        let b = backend();
        let r = b.add(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(r, vec![4.0, 6.0]);
    }

    #[test]
    fn test_mul() {
        let b = backend();
        let r = b.mul(&[2.0, 3.0], &[4.0, 5.0]).unwrap();
        assert_eq!(r, vec![8.0, 15.0]);
    }

    #[test]
    fn test_scale() {
        let b = backend();
        let r = b.scale(&[1.0, 2.0, 3.0], 2.0).unwrap();
        assert_eq!(r, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_silu() {
        let b = backend();
        let r = b.silu(&[0.0]).unwrap();
        // silu(0) = 0 / (1 + 1) = 0
        assert!((r[0] - 0.0).abs() < 1e-6);

        let r2 = b.silu(&[1.0]).unwrap();
        // silu(1) = 1 / (1 + exp(-1)) ~= 0.7310586
        assert!((r2[0] - 0.7310586).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let b = backend();
        let r = b.softmax(&[1.0, 2.0, 3.0], 3).unwrap();
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Values should be monotonically increasing
        assert!(r[0] < r[1]);
        assert!(r[1] < r[2]);
    }

    #[test]
    fn test_rms_norm() {
        let b = backend();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let r = b.rms_norm(&x, &w, 1e-5, 4).unwrap();
        // rms = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + eps) ~= 2.7386
        // normalized: [1/rms, 2/rms, 3/rms, 4/rms]
        let rms = (7.5f32 + 1e-5).sqrt();
        assert!((r[0] - 1.0 / rms).abs() < 1e-5);
        assert!((r[1] - 2.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_rope_zero_pos() {
        let b = backend();
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let (q_out, k_out) = b.rope(&q, &k, 4, 0, 1, 1).unwrap();
        // At pos=0, theta=0 for all pairs, so cos=1, sin=0 => no rotation
        assert!((q_out[0] - 1.0).abs() < 1e-6);
        assert!((q_out[1] - 0.0).abs() < 1e-6);
        assert!((k_out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_length_mismatch() {
        let b = backend();
        assert!(b.add(&[1.0], &[1.0, 2.0]).is_err());
    }
}
