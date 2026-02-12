use std::io::{BufReader, Seek};
use std::path::Path;

use memmap2::Mmap;

use ir_tensor::{DType, Shape, Tensor};

use crate::error::{ModelError, Result};
use super::header::{GgufHeader, GGUF_DEFAULT_ALIGNMENT};
use super::metadata::GgufMetadata;
use super::tensor_info::{self, GgufTensorInfo};

/// A parsed GGUF file backed by a memory-mapped region.
///
/// After parsing the header, metadata, and tensor info table from the file,
/// the entire file is memory-mapped so that tensor data can be accessed
/// without additional reads.
pub struct GgufFile {
    /// Parsed header (version, tensor/KV counts).
    pub header: GgufHeader,
    /// Parsed metadata key-value entries.
    pub metadata: GgufMetadata,
    /// Parsed tensor info entries (name, shape, dtype, offset).
    pub tensor_infos: Vec<GgufTensorInfo>,
    /// Memory-mapped file contents.
    mmap: Mmap,
    /// Byte offset within the file where tensor data begins (aligned).
    data_offset: usize,
}

impl GgufFile {
    /// Open and parse a GGUF file from disk.
    ///
    /// This reads the header, metadata, and tensor info table sequentially
    /// using buffered I/O, then memory-maps the entire file so tensor data
    /// can be accessed via slices.
    pub fn open(path: &Path) -> Result<GgufFile> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(&file);

        let header = GgufHeader::parse(&mut reader)?;
        let metadata = GgufMetadata::parse_kv(&mut reader, header.n_kv)?;
        let tensor_infos = tensor_info::parse_tensor_infos(&mut reader, header.n_tensors)?;

        // Determine current position in the file (end of tensor info table).
        let current_pos = reader.stream_position()? as usize;

        // Align to GGUF_DEFAULT_ALIGNMENT to find where tensor data starts.
        let data_offset = (current_pos + GGUF_DEFAULT_ALIGNMENT - 1)
            & !(GGUF_DEFAULT_ALIGNMENT - 1);

        // Memory-map the entire file.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(GgufFile {
            header,
            metadata,
            tensor_infos,
            mmap,
            data_offset,
        })
    }

    /// Get a raw byte slice for a tensor's data within the memory-mapped file.
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> &[u8] {
        let start = self.data_offset + info.offset as usize;
        let size = info.data_size();
        &self.mmap[start..start + size]
    }

    /// Load a tensor by name, dequantizing to f32 if needed.
    ///
    /// Supports F32, F16, Q4_0, and Q8_0 formats.
    pub fn get_tensor_f32(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensor_infos
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| ModelError::TensorNotFound(name.to_string()))?;

        let raw = self.tensor_data(info);
        let numel = info.numel();
        let shape_dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();

        let data = match info.dtype {
            DType::F32 => dequantize_f32(raw, numel),
            DType::F16 => dequantize_f16(raw, numel),
            DType::Q4_0 => dequantize_q4_0(raw, numel),
            DType::Q8_0 => dequantize_q8_0(raw, numel),
        };

        Ok(Tensor::new(data, Shape::new(shape_dims)))
    }
}

/// Reinterpret raw bytes as f32 values (little-endian).
fn dequantize_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let offset = i * 4;
        let bytes: [u8; 4] = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ];
        out.push(f32::from_le_bytes(bytes));
    }
    out
}

/// Convert f16 values to f32.
fn dequantize_f16(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let offset = i * 2;
        let bytes: [u8; 2] = [data[offset], data[offset + 1]];
        let h = half::f16::from_le_bytes(bytes);
        out.push(h.to_f32());
    }
    out
}

/// Dequantize Q4_0 blocks to f32.
///
/// Q4_0 block layout (18 bytes total, 32 elements per block):
///   - 2 bytes: f16 scale factor
///   - 16 bytes: 32 packed 4-bit values (2 per byte, lower nibble first)
///
/// Each 4-bit value is unsigned (0..15); dequantized as: (nibble - 8) * scale.
fn dequantize_q4_0(data: &[u8], numel: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 (scale) + 16 (nibbles)

    let n_blocks = numel.div_ceil(BLOCK_SIZE);
    let mut out = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;

        // Read f16 scale.
        let scale_bytes: [u8; 2] = [data[block_start], data[block_start + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Read 16 bytes of packed nibbles (32 values).
        for byte_idx in 0..16 {
            let byte = data[block_start + 2 + byte_idx];

            // Lower nibble first.
            let lo = (byte & 0x0F) as i32 - 8;
            out.push(lo as f32 * scale);

            // Upper nibble second.
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push(hi as f32 * scale);
        }
    }

    // Trim to exact element count (last block may have padding).
    out.truncate(numel);
    out
}

/// Dequantize Q8_0 blocks to f32.
///
/// Q8_0 block layout (34 bytes total, 32 elements per block):
///   - 2 bytes: f16 scale factor
///   - 32 bytes: 32 signed 8-bit values
///
/// Dequantized as: value * scale.
fn dequantize_q8_0(data: &[u8], numel: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (scale) + 32 (quants)

    let n_blocks = numel.div_ceil(BLOCK_SIZE);
    let mut out = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_BYTES;

        // Read f16 scale.
        let scale_bytes: [u8; 2] = [data[block_start], data[block_start + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Read 32 signed 8-bit values.
        for i in 0..BLOCK_SIZE {
            let val = data[block_start + 2 + i] as i8;
            out.push(val as f32 * scale);
        }
    }

    // Trim to exact element count (last block may have padding).
    out.truncate(numel);
    out
}
