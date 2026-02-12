/// Status codes returned by all FFI functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IRStatus {
    Ok = 0,
    ErrorInvalidArgument = 1,
    ErrorModelLoad = 2,
    ErrorGenerate = 3,
    ErrorOutOfMemory = 4,
    ErrorInternal = 5,
}

/// Compute backend type selector.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum IRBackendType {
    Cpu = 0,
    Metal = 1,
}

/// Parameters controlling text generation.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct IRGenerateParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for IRGenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            seed: 0,
        }
    }
}

/// Callback for streaming token output.
/// Returns true to continue generation, false to stop.
pub type IRStreamCallback = Option<
    extern "C" fn(
        token: *const std::os::raw::c_char,
        user_data: *mut std::os::raw::c_void,
    ) -> bool,
>;
