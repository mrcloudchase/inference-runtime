use crate::backend::ComputeBackend;
use crate::dtype::DType;
use crate::error::{Result, TensorError};
use crate::shape::Shape;
use crate::storage::CpuStorage;

/// A tensor backed by CPU storage.
///
/// Holds contiguous, row-major f32 data with an associated shape and dtype.
/// Operations that require computation are dispatched to a `ComputeBackend`.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: CpuStorage,
    shape: Shape,
    dtype: DType,
}

impl Tensor {
    /// Create a new tensor from f32 data and a shape.
    ///
    /// # Panics
    /// Panics if `data.len() != shape.numel()`.
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.numel(),
            "data length {} does not match shape {:?} (numel={})",
            data.len(),
            shape,
            shape.numel()
        );
        Tensor {
            storage: CpuStorage::from_f32_vec(data),
            shape,
            dtype: DType::F32,
        }
    }

    /// Create a zero-filled tensor with the given shape.
    pub fn zeros(shape: Shape) -> Self {
        let n = shape.numel();
        Tensor {
            storage: CpuStorage::from_f32_vec(vec![0.0; n]),
            shape,
            dtype: DType::F32,
        }
    }

    /// Create a tensor filled with ones with the given shape.
    pub fn ones(shape: Shape) -> Self {
        let n = shape.numel();
        Tensor {
            storage: CpuStorage::from_f32_vec(vec![1.0; n]),
            shape,
            dtype: DType::F32,
        }
    }

    /// Returns a reference to the tensor's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the tensor's data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the underlying data as an f32 slice.
    ///
    /// # Panics
    /// Panics if the storage is not F32 (should not happen in Phase 1).
    pub fn data_f32(&self) -> &[f32] {
        self.storage
            .as_f32_slice()
            .expect("tensor storage is not F32")
    }

    /// Reshape the tensor, returning a new tensor with the same data but
    /// a different shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(&self, new_shape: Shape) -> Result<Tensor> {
        if self.shape.numel() != new_shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: new_shape.dims().to_vec(),
            });
        }
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    /// Matrix multiplication of two 2D tensors using the given backend.
    ///
    /// self is [m, k], other is [k, n], result is [m, n].
    pub fn matmul(&self, other: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        if self.shape.ndim() != 2 || other.shape.ndim() != 2 {
            return Err(TensorError::Other(
                "matmul requires 2D tensors".to_string(),
            ));
        }

        let m = self.shape.dim(0);
        let k = self.shape.dim(1);
        let k2 = other.shape.dim(0);
        let n = other.shape.dim(1);

        if k != k2 {
            return Err(TensorError::MatmulMismatch { m, k, k2, n });
        }

        let result_data = backend.matmul(self.data_f32(), other.data_f32(), m, k, n)?;
        Ok(Tensor::new(result_data, Shape::new(vec![m, n])))
    }

    /// Returns the underlying storage reference.
    pub fn storage(&self) -> &CpuStorage {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    #[test]
    fn test_new_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        assert_eq!(t.shape().ndim(), 2);
        assert_eq!(t.shape().dim(0), 2);
        assert_eq!(t.shape().dim(1), 3);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.data_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_zeros_ones() {
        let z = Tensor::zeros(Shape::new(vec![2, 3]));
        assert_eq!(z.data_f32(), &[0.0; 6]);

        let o = Tensor::ones(Shape::new(vec![3]));
        assert_eq!(o.data_f32(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let r = t.reshape(Shape::new(vec![3, 2])).unwrap();
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(r.data_f32(), t.data_f32());
    }

    #[test]
    fn test_reshape_mismatch() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        assert!(t.reshape(Shape::new(vec![2, 2])).is_err());
    }

    #[test]
    #[should_panic]
    fn test_new_shape_mismatch_panics() {
        let _t = Tensor::new(vec![1.0, 2.0], Shape::new(vec![3]));
    }

    #[test]
    fn test_matmul() {
        let backend = CpuBackend::new();
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
        let c = a.matmul(&b, &backend).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.data_f32(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let backend = CpuBackend::new();
        let a = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        assert!(a.matmul(&b, &backend).is_err());
    }
}
