mod types;
mod error;
mod context;
mod streaming;

pub use types::*;
pub use error::*;
pub use context::*;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

// Import the ModelArchitecture trait so its methods (forward, reset_cache) are available.
use ir_model::ModelArchitecture;

/// Execute a closure that returns an `IRStatus`, catching any panics
/// and converting them into `IRStatus::ErrorInternal`.
fn catch_panic<F: FnOnce() -> IRStatus + std::panic::UnwindSafe>(f: F) -> IRStatus {
    match std::panic::catch_unwind(f) {
        Ok(status) => status,
        Err(_) => {
            set_last_error("internal panic".to_string());
            IRStatus::ErrorInternal
        }
    }
}

/// Create a new inference context.
///
/// On success, writes a heap-allocated `IRContext` pointer into `*ctx_out`
/// and returns `IRStatus::Ok`. The caller must later call `ir_context_destroy`
/// to free the context.
///
/// # Safety
///
/// `ctx_out` must be a valid, non-null pointer to a `*mut IRContext`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ir_context_create(
    _backend: IRBackendType,
    ctx_out: *mut *mut IRContext,
) -> IRStatus {
    catch_panic(|| {
        if ctx_out.is_null() {
            set_last_error("ctx_out is null".to_string());
            return IRStatus::ErrorInvalidArgument;
        }
        let ctx = Box::new(IRContext::new());
        unsafe {
            *ctx_out = Box::into_raw(ctx);
        }
        IRStatus::Ok
    })
}

/// Destroy a context previously created by `ir_context_create`.
///
/// Passing a null pointer is a no-op and returns `IRStatus::Ok`.
///
/// # Safety
///
/// `ctx` must be a pointer returned by `ir_context_create`, or null.
/// Must not be called twice on the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ir_context_destroy(ctx: *mut IRContext) -> IRStatus {
    if ctx.is_null() {
        return IRStatus::Ok;
    }
    unsafe { drop(Box::from_raw(ctx)) };
    IRStatus::Ok
}

/// Load a GGUF model and its tokenizer from disk.
///
/// The model file at `model_path` is opened, parsed, and loaded into
/// the context. Both the model weights and BPE tokenizer are extracted
/// from the GGUF file.
///
/// # Safety
///
/// `ctx` must be a valid pointer from `ir_context_create`.
/// `model_path` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ir_model_load(
    ctx: *mut IRContext,
    model_path: *const c_char,
) -> IRStatus {
    catch_panic(|| {
        if ctx.is_null() || model_path.is_null() {
            set_last_error("null argument".to_string());
            return IRStatus::ErrorInvalidArgument;
        }
        let ctx = unsafe { &mut *ctx };
        let path_str = match unsafe { CStr::from_ptr(model_path) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("invalid path: {}", e));
                return IRStatus::ErrorInvalidArgument;
            }
        };

        let path = Path::new(path_str);
        let gguf = match ir_model::gguf::GgufFile::open(path) {
            Ok(g) => g,
            Err(e) => {
                set_last_error(format!("failed to open GGUF: {}", e));
                return IRStatus::ErrorModelLoad;
            }
        };

        let tokenizer =
            match ir_model::tokenizer::bpe::BpeTokenizer::from_gguf(&gguf.metadata) {
                Ok(t) => t,
                Err(e) => {
                    set_last_error(format!("failed to load tokenizer: {}", e));
                    return IRStatus::ErrorModelLoad;
                }
            };

        let model =
            match ir_model::llama::LlamaModel::from_gguf(&gguf, ctx.backend.as_ref()) {
                Ok(m) => m,
                Err(e) => {
                    set_last_error(format!("failed to load model: {}", e));
                    return IRStatus::ErrorModelLoad;
                }
            };

        ctx.model = Some(model);
        ctx.tokenizer = Some(tokenizer);
        IRStatus::Ok
    })
}

/// Generate text from a prompt (non-streaming).
///
/// On success, writes a heap-allocated C string into `*output`.
/// The caller must later call `ir_free_string` to free the output string.
///
/// # Safety
///
/// `ctx` must be a valid pointer from `ir_context_create` with a loaded model.
/// `prompt` must be a valid null-terminated C string.
/// `output` must be a valid, non-null pointer to a `*mut c_char`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ir_generate(
    ctx: *mut IRContext,
    prompt: *const c_char,
    params: IRGenerateParams,
    output: *mut *mut c_char,
) -> IRStatus {
    catch_panic(|| {
        if ctx.is_null() || prompt.is_null() || output.is_null() {
            set_last_error("null argument".to_string());
            return IRStatus::ErrorInvalidArgument;
        }
        let ctx = unsafe { &mut *ctx };
        let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("invalid prompt: {}", e));
                return IRStatus::ErrorInvalidArgument;
            }
        };

        let (model, tokenizer) = match (ctx.model.as_mut(), ctx.tokenizer.as_ref()) {
            (Some(m), Some(t)) => (m, t),
            _ => {
                set_last_error("model not loaded".to_string());
                return IRStatus::ErrorGenerate;
            }
        };

        // Tokenize the prompt.
        let tokens = tokenizer.encode(prompt_str);

        // Build the sampler chain.
        let chain = ir_sampler::SamplerChain::new()
            .with(Box::new(ir_sampler::RepetitionPenaltySampler::new(
                params.repetition_penalty,
                64,
            )))
            .with(Box::new(ir_sampler::TemperatureSampler::new(
                params.temperature,
            )))
            .with(Box::new(ir_sampler::TopKSampler::new(
                params.top_k as usize,
            )))
            .with(Box::new(ir_sampler::TopPSampler::new(params.top_p)))
            .with(Box::new(ir_sampler::GreedySampler));

        // Generate tokens autoregressively.
        let mut generated = Vec::new();
        let backend = ctx.backend.as_ref();

        // Prefill: process all prompt tokens at once, starting at position 0.
        let logits = match model.forward(&tokens, 0, backend) {
            Ok(l) => l,
            Err(e) => {
                set_last_error(format!("forward pass failed: {}", e));
                return IRStatus::ErrorGenerate;
            }
        };
        let mut cur_pos = tokens.len();

        let mut next_token = chain.sample(&logits);
        if next_token == tokenizer.vocab.eos_id {
            let text = tokenizer.decode(&generated);
            match CString::new(text) {
                Ok(c) => {
                    unsafe { *output = c.into_raw() };
                    return IRStatus::Ok;
                }
                Err(e) => {
                    set_last_error(format!("output encoding error: {}", e));
                    return IRStatus::ErrorGenerate;
                }
            }
        }
        generated.push(next_token);

        // Decode: generate one token at a time.
        for _ in 1..params.max_tokens {
            let logits = match model.forward(&[next_token], cur_pos, backend) {
                Ok(l) => l,
                Err(e) => {
                    set_last_error(format!("forward pass failed: {}", e));
                    return IRStatus::ErrorGenerate;
                }
            };
            cur_pos += 1;

            next_token = chain.sample(&logits);

            if next_token == tokenizer.vocab.eos_id {
                break;
            }

            generated.push(next_token);
        }

        let text = tokenizer.decode(&generated);
        match CString::new(text) {
            Ok(c) => {
                unsafe { *output = c.into_raw() };
                IRStatus::Ok
            }
            Err(e) => {
                set_last_error(format!("output encoding error: {}", e));
                IRStatus::ErrorGenerate
            }
        }
    })
}

/// Generate text from a prompt with streaming output.
///
/// Each generated token is passed to the `callback` function as a C string.
/// The callback should return `true` to continue generation, or `false` to stop.
///
/// # Safety
///
/// `ctx` must be a valid pointer from `ir_context_create` with a loaded model.
/// `prompt` must be a valid null-terminated C string.
/// `callback` and `user_data` must remain valid for the duration of generation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ir_generate_streaming(
    ctx: *mut IRContext,
    prompt: *const c_char,
    params: IRGenerateParams,
    callback: IRStreamCallback,
    user_data: *mut std::os::raw::c_void,
) -> IRStatus {
    catch_panic(|| {
        if ctx.is_null() || prompt.is_null() {
            set_last_error("null argument".to_string());
            return IRStatus::ErrorInvalidArgument;
        }
        let ctx = unsafe { &mut *ctx };
        let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("invalid prompt: {}", e));
                return IRStatus::ErrorInvalidArgument;
            }
        };

        let (model, tokenizer) = match (ctx.model.as_mut(), ctx.tokenizer.as_ref()) {
            (Some(m), Some(t)) => (m, t),
            _ => {
                set_last_error("model not loaded".to_string());
                return IRStatus::ErrorGenerate;
            }
        };

        let tokens = tokenizer.encode(prompt_str);
        let chain = ir_sampler::SamplerChain::new()
            .with(Box::new(ir_sampler::TemperatureSampler::new(
                params.temperature,
            )))
            .with(Box::new(ir_sampler::TopKSampler::new(
                params.top_k as usize,
            )))
            .with(Box::new(ir_sampler::TopPSampler::new(params.top_p)))
            .with(Box::new(ir_sampler::GreedySampler));

        let backend = ctx.backend.as_ref();

        // Prefill: process all prompt tokens at once.
        let logits = match model.forward(&tokens, 0, backend) {
            Ok(l) => l,
            Err(e) => {
                set_last_error(format!("forward pass failed: {}", e));
                return IRStatus::ErrorGenerate;
            }
        };
        let mut cur_pos = tokens.len();

        let mut next_token = chain.sample(&logits);
        if next_token == tokenizer.vocab.eos_id {
            return IRStatus::Ok;
        }

        let text = tokenizer.decode(&[next_token]);
        if !streaming::invoke_callback(callback, user_data, &text) {
            return IRStatus::Ok;
        }

        // Decode: generate one token at a time.
        for _ in 1..params.max_tokens {
            let logits = match model.forward(&[next_token], cur_pos, backend) {
                Ok(l) => l,
                Err(e) => {
                    set_last_error(format!("forward pass failed: {}", e));
                    return IRStatus::ErrorGenerate;
                }
            };
            cur_pos += 1;

            next_token = chain.sample(&logits);
            if next_token == tokenizer.vocab.eos_id {
                break;
            }

            let text = tokenizer.decode(&[next_token]);
            if !streaming::invoke_callback(callback, user_data, &text) {
                break; // user requested stop
            }
        }

        IRStatus::Ok
    })
}

/// Reset the model's KV cache (e.g. to start a new conversation).
///
/// # Safety
///
/// `ctx` must be a valid pointer from `ir_context_create`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ir_reset(ctx: *mut IRContext) -> IRStatus {
    if ctx.is_null() {
        return IRStatus::ErrorInvalidArgument;
    }
    let ctx = unsafe { &mut *ctx };
    if let Some(model) = ctx.model.as_mut() {
        model.reset_cache();
    }
    IRStatus::Ok
}

/// Retrieve the last error message.
///
/// Returns a pointer to a C string describing the most recent error, or
/// null if no error has occurred. The caller must free the returned string
/// with `ir_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn ir_last_error() -> *const c_char {
    match error::take_last_error() {
        Some(e) => e.into_raw(),
        None => std::ptr::null(),
    }
}

/// Free a string previously returned by `ir_generate` or `ir_last_error`.
///
/// # Safety
///
/// `s` must be a pointer returned by `ir_generate`, `ir_last_error`, or null.
/// Must not be called twice on the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ir_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}
