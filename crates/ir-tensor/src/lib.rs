//! `ir-tensor` - Tensor library with pluggable compute backends for inference-runtime.
//!
//! This crate provides:
//! - A `Tensor` type backed by CPU storage
//! - A `ComputeBackend` trait for pluggable compute (CPU, Metal, etc.)
//! - A reference `CpuBackend` implementation
//! - Shape utilities and broadcasting
//! - Data type definitions (F32, F16, quantized formats)

pub mod backend;
pub mod cpu;
pub mod dtype;
pub mod error;
#[cfg(feature = "metal")]
pub mod metal;
pub mod shape;
pub mod storage;
pub mod tensor;

// Re-export primary types at the crate root for convenience.
pub use backend::ComputeBackend;
pub use cpu::CpuBackend;
pub use dtype::DType;
pub use error::{Result, TensorError};
pub use shape::Shape;
pub use storage::CpuStorage;
pub use tensor::Tensor;
