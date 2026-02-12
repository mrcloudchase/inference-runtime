# Roadmap

This document tracks the development phases and planned features for inference-runtime.

## Phase 1: CPU Inference End-to-End (Complete)

The foundation — load a GGUF model, tokenize, run the transformer forward pass, sample, and decode on CPU.

- [x] **ir-tensor**: Tensor struct, Shape, DType (F32/F16/Q4_0/Q8_0), CpuStorage
- [x] **ir-tensor**: ComputeBackend trait with CpuBackend (matmul, add, mul, scale, rms_norm, softmax, rope, silu)
- [x] **ir-model**: GGUF v3 binary parser with mmap-backed tensor access
- [x] **ir-model**: Dequantization (F32, F16, Q4_0, Q8_0 to f32)
- [x] **ir-model**: BPE tokenizer from GGUF metadata (encode + decode)
- [x] **ir-model**: LLaMA architecture (GQA attention, SwiGLU FFN, RoPE, KV cache)
- [x] **ir-sampler**: Temperature, Top-K, Top-P, repetition penalty, greedy, distribution sampling
- [x] **ir-sampler**: SamplerChain composition
- [x] **ir-ffi**: C API with cbindgen, thread-local errors, panic catching
- [x] **Go wrapper**: CGo bindings with streaming callback bridge
- [x] **Go wrapper**: CLI commands (run, serve, list, info) with Cobra
- [x] **Go wrapper**: REST API (/api/generate, /api/chat, /api/tags, /api/delete, /api/health)
- [x] **Go wrapper**: Local model registry with JSON manifests

## Phase 2: Metal GPU Acceleration

Accelerate inference on Apple Silicon using Metal compute shaders. Target 10-50x speedup over CPU.

- [ ] MetalBackend implementing ComputeBackend trait
- [ ] Metal device/command queue initialization (objc2-metal)
- [ ] MSL compute shaders for core ops (matmul, rms_norm, softmax, rope, silu)
- [ ] GPU buffer management and CPU-GPU data transfer
- [ ] GPU-resident KV cache to avoid per-token transfers
- [ ] Backend selection via CLI flag (`ir run --backend metal`)
- [ ] Backend selection via API parameter
- [ ] Benchmarks: CPU vs Metal tokens/sec for various model sizes

## Phase 3: Performance Optimization

Improve throughput and memory efficiency without changing the architecture.

- [ ] Quantized matmul (Q4_0/Q8_0 compute without full dequantization)
- [ ] SIMD-accelerated CPU ops (ARM NEON for matmul, activation functions)
- [ ] Tiled/blocked matmul for better cache locality
- [ ] Memory-mapped weight access (avoid dequantizing all weights into RAM at load)
- [ ] Batch prefill (process multiple prompt tokens in a single matmul)
- [ ] KV cache memory optimization (only allocate for actual sequence length)
- [ ] Token generation throughput benchmarking and profiling

## Phase 4: Additional Architectures

Support more model families beyond LLaMA.

- [ ] Mistral (sliding window attention)
- [ ] Phi (partial rotary embedding, dense attention)
- [ ] Gemma (GeGLU activation, different norm placement)
- [ ] Architecture auto-detection from GGUF metadata (`general.architecture` key)
- [ ] Shared weight loading infrastructure across architectures

## Phase 5: Model Management

Full model lifecycle management like ollama.

- [ ] `ir pull <model>` — download GGUF models from HuggingFace
- [ ] Progress bar for model downloads
- [ ] SHA256 integrity verification for downloaded models
- [ ] Blob storage (content-addressed by hash, deduplication)
- [ ] Model aliases and tagging
- [ ] `ir rm <model>` — remove downloaded models and free disk space
- [ ] `ir cp <src> <dst>` — copy/alias a model

## Phase 6: Production Features

Features needed for production deployment.

- [ ] OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/completions`)
- [ ] Concurrent request handling (multiple inference contexts)
- [ ] Request queuing and backpressure
- [ ] Prompt template support (chat templates from GGUF metadata)
- [ ] Token count / context window management
- [ ] Structured logging (JSON format)
- [ ] Prometheus metrics endpoint
- [ ] Docker container image
- [ ] CUDA backend (Linux GPU support)
- [ ] Vulkan backend (cross-platform GPU)

## Future Ideas

Exploratory features not yet planned for a specific phase.

- Multi-model serving (hot-swap between models)
- Speculative decoding
- Continuous batching
- LoRA adapter loading
- Embedding extraction endpoint
- Function calling / tool use support
- Vision model support (multimodal)
- WebAssembly backend for browser inference
