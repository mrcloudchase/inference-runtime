use std::fmt;

/// Supported data types for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (IEEE 754 half-precision, via the `half` crate).
    F16,
    /// 4-bit quantized format (GGUF Q4_0 block type).
    Q4_0,
    /// 8-bit quantized format (GGUF Q8_0 block type).
    Q8_0,
}

impl DType {
    /// Returns the size in bytes of a single element for non-quantized types,
    /// or the block size for quantized types.
    ///
    /// - F32: 4 bytes per element
    /// - F16: 2 bytes per element (using `half::f16`)
    /// - Q4_0: 18 bytes per block of 32 elements (2-byte scale + 16 bytes of nibbles)
    /// - Q8_0: 34 bytes per block of 32 elements (2-byte scale + 32 bytes of quants)
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q4_0 => 18,
            DType::Q8_0 => 34,
        }
    }

    /// Converts a GGUF type ID to a `DType`.
    ///
    /// GGUF type IDs:
    /// - 0 => F32
    /// - 1 => F16
    /// - 2 => Q4_0
    /// - 8 => Q8_0
    pub fn from_gguf_type(id: u32) -> Option<DType> {
        match id {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            8 => Some(DType::Q8_0),
            _ => None,
        }
    }

    /// Returns the GGUF type ID for this `DType`.
    pub fn to_gguf_type(&self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::Q4_0 => 2,
            DType::Q8_0 => 8,
        }
    }

    /// Returns the number of elements per quantization block, or 1 for
    /// non-quantized types.
    pub fn block_size(&self) -> usize {
        match self {
            DType::F32 | DType::F16 => 1,
            DType::Q4_0 | DType::Q8_0 => 32,
        }
    }

    /// Returns true if this dtype is a quantized format.
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Q4_0 | DType::Q8_0)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::Q4_0 => write!(f, "q4_0"),
            DType::Q8_0 => write!(f, "q8_0"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::Q4_0.size_in_bytes(), 18);
        assert_eq!(DType::Q8_0.size_in_bytes(), 34);
    }

    #[test]
    fn test_gguf_roundtrip() {
        for dtype in &[DType::F32, DType::F16, DType::Q4_0, DType::Q8_0] {
            let id = dtype.to_gguf_type();
            let back = DType::from_gguf_type(id).unwrap();
            assert_eq!(*dtype, back);
        }
    }

    #[test]
    fn test_gguf_unknown() {
        assert!(DType::from_gguf_type(999).is_none());
    }
}
