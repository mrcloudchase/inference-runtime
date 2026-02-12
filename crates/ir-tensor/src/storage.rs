use crate::dtype::DType;
use crate::error::{Result, TensorError};

/// CPU-side tensor storage.
///
/// Phase 1 focuses on F32 storage only. Additional variants (F16, quantized)
/// will be added in later phases.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    /// 32-bit floating point storage.
    F32(Vec<f32>),
}

impl CpuStorage {
    /// Number of elements in this storage.
    pub fn len(&self) -> usize {
        match self {
            CpuStorage::F32(v) => v.len(),
        }
    }

    /// Returns true if the storage contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the data as an f32 slice.
    ///
    /// # Errors
    /// Returns an error if the storage is not F32.
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        match self {
            CpuStorage::F32(v) => Ok(v.as_slice()),
        }
    }

    /// Returns the data as a mutable f32 slice.
    ///
    /// # Errors
    /// Returns an error if the storage is not F32.
    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32]> {
        match self {
            CpuStorage::F32(v) => Ok(v.as_mut_slice()),
        }
    }

    /// Create zero-filled storage for the given dtype and element count.
    ///
    /// # Errors
    /// Returns an error for unsupported dtypes (Phase 1: only F32 is supported).
    pub fn zeros(dtype: DType, n: usize) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(CpuStorage::F32(vec![0.0; n])),
            other => Err(TensorError::UnsupportedDType(format!(
                "{} storage not yet implemented",
                other
            ))),
        }
    }

    /// Create storage from an f32 vector.
    pub fn from_f32_vec(data: Vec<f32>) -> Self {
        CpuStorage::F32(data)
    }

    /// Returns the dtype of this storage.
    pub fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f32_vec() {
        let s = CpuStorage::from_f32_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zeros_f32() {
        let s = CpuStorage::zeros(DType::F32, 5).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.as_f32_slice().unwrap(), &[0.0; 5]);
    }

    #[test]
    fn test_zeros_unsupported() {
        assert!(CpuStorage::zeros(DType::F16, 5).is_err());
        assert!(CpuStorage::zeros(DType::Q4_0, 5).is_err());
    }

    #[test]
    fn test_dtype() {
        let s = CpuStorage::from_f32_vec(vec![]);
        assert_eq!(s.dtype(), DType::F32);
    }

    #[test]
    fn test_mut_slice() {
        let mut s = CpuStorage::from_f32_vec(vec![1.0, 2.0]);
        let slice = s.as_f32_slice_mut().unwrap();
        slice[0] = 42.0;
        assert_eq!(s.as_f32_slice().unwrap()[0], 42.0);
    }
}
