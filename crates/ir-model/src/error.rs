use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GGUF magic: expected 'GGUF', got {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("missing metadata key: {0}")]
    MissingKey(String),
    #[error("type mismatch for key '{key}': expected {expected}, got {got}")]
    TypeMismatch {
        key: String,
        expected: String,
        got: String,
    },
    #[error("unsupported GGUF type ID: {0}")]
    UnsupportedGgufType(u32),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("tokenizer error: {0}")]
    TokenizerError(String),
    #[error("tensor error: {0}")]
    TensorError(#[from] ir_tensor::TensorError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;
