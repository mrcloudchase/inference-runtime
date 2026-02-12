use std::io::Read;

use ir_tensor::DType;

use crate::error::{ModelError, Result};

/// Describes a single tensor stored within a GGUF file.
pub struct GgufTensorInfo {
    /// Tensor name (e.g. "blk.0.attn_q.weight").
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Size of each dimension.
    pub dims: Vec<u64>,
    /// Data type of the stored tensor data.
    pub dtype: DType,
    /// Byte offset of this tensor's data from the start of the tensor data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements in this tensor.
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Compute the total byte size of this tensor's raw data in the file.
    pub fn data_size(&self) -> usize {
        let numel = self.numel();
        let block_size = self.dtype.block_size();
        let n_blocks = numel.div_ceil(block_size);
        n_blocks * self.dtype.size_in_bytes()
    }
}

/// Read a GGUF-encoded string: u64 length then that many bytes.
fn read_gguf_string(reader: &mut impl Read) -> Result<String> {
    let mut buf8 = [0u8; 8];
    reader.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| ModelError::Other(format!("invalid UTF-8 in tensor name: {}", e)))
}

/// Parse `n_tensors` tensor info entries from a reader.
///
/// Each entry:
/// 1. GGUF string name
/// 2. u32 number of dimensions
/// 3. n_dims x u64 dimension sizes
/// 4. u32 GGUF type ID (mapped via `DType::from_gguf_type`)
/// 5. u64 byte offset within the tensor data section
pub fn parse_tensor_infos(reader: &mut impl Read, n_tensors: u64) -> Result<Vec<GgufTensorInfo>> {
    let mut infos = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_gguf_string(reader)?;

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let n_dims = u32::from_le_bytes(buf4);

        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let mut buf8 = [0u8; 8];
            reader.read_exact(&mut buf8)?;
            dims.push(u64::from_le_bytes(buf8));
        }

        reader.read_exact(&mut buf4)?;
        let type_id = u32::from_le_bytes(buf4);
        let dtype = DType::from_gguf_type(type_id)
            .ok_or(ModelError::UnsupportedGgufType(type_id))?;

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let offset = u64::from_le_bytes(buf8);

        infos.push(GgufTensorInfo {
            name,
            n_dims,
            dims,
            dtype,
            offset,
        });
    }
    Ok(infos)
}
