use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    #[error("dtype mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: String, got: String },
    #[error("invalid axis {axis} for tensor with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },
    #[error("cannot broadcast shapes {a:?} and {b:?}")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },
    #[error("matmul dimension mismatch: [{m}x{k}] @ [{k2}x{n}]")]
    MatmulMismatch {
        m: usize,
        k: usize,
        k2: usize,
        n: usize,
    },
    #[error("unsupported dtype: {0}")]
    UnsupportedDType(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;
