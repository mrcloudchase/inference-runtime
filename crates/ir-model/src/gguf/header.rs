use std::io::Read;

use crate::error::{ModelError, Result};

/// The four-byte magic number identifying a GGUF file: ASCII "GGUF".
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Default alignment (in bytes) for tensor data within a GGUF file.
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Parsed GGUF file header.
pub struct GgufHeader {
    /// GGUF format version (we support v3).
    pub version: u32,
    /// Number of tensors stored in the file.
    pub n_tensors: u64,
    /// Number of key-value metadata entries.
    pub n_kv: u64,
}

impl GgufHeader {
    /// Parse a GGUF header from the beginning of a reader.
    ///
    /// Reads and validates the 4-byte magic, then reads the version (u32 LE),
    /// tensor count (u64 LE), and KV count (u64 LE). Only version 3 is
    /// currently supported.
    pub fn parse(reader: &mut impl Read) -> Result<GgufHeader> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != GGUF_MAGIC {
            return Err(ModelError::InvalidMagic(magic));
        }

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 3 {
            return Err(ModelError::UnsupportedVersion(version));
        }

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let n_tensors = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let n_kv = u64::from_le_bytes(buf8);

        Ok(GgufHeader {
            version,
            n_tensors,
            n_kv,
        })
    }
}
