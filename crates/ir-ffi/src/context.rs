use std::sync::Arc;
use ir_tensor::CpuBackend;
use ir_model::llama::LlamaModel;
use ir_model::tokenizer::bpe::BpeTokenizer;

/// Opaque context handle that owns the backend, model, and tokenizer.
pub struct IRContext {
    pub backend: Arc<CpuBackend>,
    pub model: Option<LlamaModel>,
    pub tokenizer: Option<BpeTokenizer>,
}

impl Default for IRContext {
    fn default() -> Self {
        Self::new()
    }
}

impl IRContext {
    pub fn new() -> Self {
        Self {
            backend: Arc::new(CpuBackend::new()),
            model: None,
            tokenizer: None,
        }
    }
}
