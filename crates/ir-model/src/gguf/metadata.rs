use std::collections::HashMap;
use std::io::Read;

use crate::error::{ModelError, Result};

/// A single GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
}

impl GgufMetadataValue {
    /// Returns a human-readable name for the variant (used in error messages).
    fn type_name(&self) -> &'static str {
        match self {
            GgufMetadataValue::U8(_) => "U8",
            GgufMetadataValue::I8(_) => "I8",
            GgufMetadataValue::U16(_) => "U16",
            GgufMetadataValue::I16(_) => "I16",
            GgufMetadataValue::U32(_) => "U32",
            GgufMetadataValue::I32(_) => "I32",
            GgufMetadataValue::U64(_) => "U64",
            GgufMetadataValue::I64(_) => "I64",
            GgufMetadataValue::F32(_) => "F32",
            GgufMetadataValue::F64(_) => "F64",
            GgufMetadataValue::Bool(_) => "Bool",
            GgufMetadataValue::String(_) => "String",
            GgufMetadataValue::Array(_) => "Array",
        }
    }
}

/// Collection of GGUF metadata key-value pairs.
pub struct GgufMetadata {
    pub entries: HashMap<String, GgufMetadataValue>,
}

impl GgufMetadata {
    /// Retrieve a string value by key.
    pub fn get_string(&self, key: &str) -> Result<&str> {
        match self.entries.get(key) {
            Some(GgufMetadataValue::String(s)) => Ok(s.as_str()),
            Some(other) => Err(ModelError::TypeMismatch {
                key: key.to_string(),
                expected: "String".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ModelError::MissingKey(key.to_string())),
        }
    }

    /// Retrieve a u32 value by key.
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        match self.entries.get(key) {
            Some(GgufMetadataValue::U32(v)) => Ok(*v),
            Some(other) => Err(ModelError::TypeMismatch {
                key: key.to_string(),
                expected: "U32".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ModelError::MissingKey(key.to_string())),
        }
    }

    /// Retrieve a u64 value by key.
    pub fn get_u64(&self, key: &str) -> Result<u64> {
        match self.entries.get(key) {
            Some(GgufMetadataValue::U64(v)) => Ok(*v),
            Some(other) => Err(ModelError::TypeMismatch {
                key: key.to_string(),
                expected: "U64".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ModelError::MissingKey(key.to_string())),
        }
    }

    /// Retrieve an i32 value by key.
    pub fn get_i32(&self, key: &str) -> Result<i32> {
        match self.entries.get(key) {
            Some(GgufMetadataValue::I32(v)) => Ok(*v),
            Some(other) => Err(ModelError::TypeMismatch {
                key: key.to_string(),
                expected: "I32".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ModelError::MissingKey(key.to_string())),
        }
    }

    /// Retrieve an f32 value by key.
    pub fn get_f32(&self, key: &str) -> Result<f32> {
        match self.entries.get(key) {
            Some(GgufMetadataValue::F32(v)) => Ok(*v),
            Some(other) => Err(ModelError::TypeMismatch {
                key: key.to_string(),
                expected: "F32".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ModelError::MissingKey(key.to_string())),
        }
    }

    /// Retrieve a string array value by key.
    pub fn get_string_array(&self, key: &str) -> Result<Vec<String>> {
        match self.entries.get(key) {
            Some(GgufMetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for (i, v) in arr.iter().enumerate() {
                    match v {
                        GgufMetadataValue::String(s) => result.push(s.clone()),
                        other => {
                            return Err(ModelError::TypeMismatch {
                                key: format!("{}[{}]", key, i),
                                expected: "String".to_string(),
                                got: other.type_name().to_string(),
                            });
                        }
                    }
                }
                Ok(result)
            }
            Some(other) => Err(ModelError::TypeMismatch {
                key: key.to_string(),
                expected: "Array".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ModelError::MissingKey(key.to_string())),
        }
    }

    /// Retrieve an f32 array value by key.
    pub fn get_f32_array(&self, key: &str) -> Result<Vec<f32>> {
        match self.entries.get(key) {
            Some(GgufMetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for (i, v) in arr.iter().enumerate() {
                    match v {
                        GgufMetadataValue::F32(f) => result.push(*f),
                        other => {
                            return Err(ModelError::TypeMismatch {
                                key: format!("{}[{}]", key, i),
                                expected: "F32".to_string(),
                                got: other.type_name().to_string(),
                            });
                        }
                    }
                }
                Ok(result)
            }
            Some(other) => Err(ModelError::TypeMismatch {
                key: key.to_string(),
                expected: "Array".to_string(),
                got: other.type_name().to_string(),
            }),
            None => Err(ModelError::MissingKey(key.to_string())),
        }
    }

    /// Parse `n_kv` key-value metadata entries from a reader.
    ///
    /// Each entry consists of:
    /// 1. A GGUF string key (u64 length + UTF-8 bytes).
    /// 2. A u32 value type ID.
    /// 3. The value payload, whose format depends on the type ID.
    ///
    /// GGUF value type IDs:
    ///   0=U8, 1=I8, 2=U16, 3=I16, 4=U32, 5=I32, 6=F32, 7=Bool,
    ///   8=String, 9=Array, 10=U64, 11=I64, 12=F64
    pub fn parse_kv(reader: &mut impl Read, n_kv: u64) -> Result<GgufMetadata> {
        let mut entries = HashMap::new();
        for _ in 0..n_kv {
            let key = read_gguf_string(reader)?;
            let mut buf4 = [0u8; 4];
            reader.read_exact(&mut buf4)?;
            let type_id = u32::from_le_bytes(buf4);
            let value = read_value(reader, type_id)?;
            entries.insert(key, value);
        }
        Ok(GgufMetadata { entries })
    }
}

/// Read a GGUF string: u64 length followed by that many UTF-8 bytes.
fn read_gguf_string(reader: &mut impl Read) -> Result<String> {
    let mut buf8 = [0u8; 8];
    reader.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| ModelError::Other(format!("invalid UTF-8 in string: {}", e)))
}

/// Read a single GGUF metadata value given its type ID.
fn read_value(reader: &mut impl Read, type_id: u32) -> Result<GgufMetadataValue> {
    match type_id {
        0 => {
            // U8
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::U8(buf[0]))
        }
        1 => {
            // I8
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::I8(buf[0] as i8))
        }
        2 => {
            // U16
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::U16(u16::from_le_bytes(buf)))
        }
        3 => {
            // I16
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::I16(i16::from_le_bytes(buf)))
        }
        4 => {
            // U32
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::U32(u32::from_le_bytes(buf)))
        }
        5 => {
            // I32
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::I32(i32::from_le_bytes(buf)))
        }
        6 => {
            // F32
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::F32(f32::from_le_bytes(buf)))
        }
        7 => {
            // Bool
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::Bool(buf[0] != 0))
        }
        8 => {
            // String
            let s = read_gguf_string(reader)?;
            Ok(GgufMetadataValue::String(s))
        }
        9 => {
            // Array: u32 element_type, u64 count, then count values of element_type
            let mut buf4 = [0u8; 4];
            reader.read_exact(&mut buf4)?;
            let elem_type = u32::from_le_bytes(buf4);

            let mut buf8 = [0u8; 8];
            reader.read_exact(&mut buf8)?;
            let count = u64::from_le_bytes(buf8) as usize;

            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_value(reader, elem_type)?);
            }
            Ok(GgufMetadataValue::Array(values))
        }
        10 => {
            // U64
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::U64(u64::from_le_bytes(buf)))
        }
        11 => {
            // I64
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::I64(i64::from_le_bytes(buf)))
        }
        12 => {
            // F64
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::F64(f64::from_le_bytes(buf)))
        }
        other => Err(ModelError::UnsupportedGgufType(other)),
    }
}
