# AGENTS.md

Always reference [ROADMAP.md](./ROADMAP.md) when planning or implementing new features to understand project phases, priorities, and what has already been completed.

## Build

```sh
make            # full build: Rust release crates then Go binary via CGo
make rust       # cargo build --release (builds ir-tensor, ir-sampler, ir-model, ir-ffi)
make go         # Go binary (requires Rust build first); output at target/release/ir
make test       # cargo test --workspace && cd go-wrapper && go test ./...
make check      # cargo clippy --workspace -- -D warnings
make fmt        # cargo fmt --all && gofmt -w go-wrapper/
```

Single Rust crate test: `cargo test -p ir-tensor test_name`

Go tests require CGo environment variables since the Go binary links against the Rust `libir_ffi`:
```sh
cd go-wrapper && CGO_ENABLED=1 GOARCH=arm64 \
  CGO_LDFLAGS="-L../target/release -lir_ffi" \
  CGO_CFLAGS="-I./bindings" go test ./...
```

Platform: macOS Apple Silicon only. Go build must use `GOARCH=arm64` and `CGO_ENABLED=1` (the installed Go may be an x86_64 binary running under Rosetta, but Rust compiles native arm64).

## File Structure

```
inference-runtime/
├── Cargo.toml                          # Rust workspace root (resolver=2, 4 members)
├── Makefile                            # Orchestrates Rust + Go builds
│
├── crates/
│   ├── ir-tensor/                      # Tensor library + compute backends
│   │   └── src/
│   │       ├── lib.rs                  # Re-exports: Tensor, Shape, DType, CpuBackend, ComputeBackend
│   │       ├── dtype.rs                # DType enum (F32, F16, Q4_0, Q8_0) + GGUF type ID conversion
│   │       ├── shape.rs                # Shape: dims, strides, numel, numpy-style broadcasting
│   │       ├── storage.rs              # CpuStorage enum (F32 variant only), f32 slice access
│   │       ├── tensor.rs               # Tensor: new/zeros/ones, reshape, matmul dispatch
│   │       ├── backend.rs              # ComputeBackend trait (matmul/add/mul/scale/rms_norm/softmax/rope/silu)
│   │       ├── error.rs                # TensorError enum (ShapeMismatch, MatmulMismatch, BroadcastError, ...)
│   │       ├── cpu/
│   │       │   ├── mod.rs              # CpuBackend: all ComputeBackend ops as naive loops
│   │       │   ├── matmul.rs           # Placeholder for future SIMD/tiled matmul optimizations
│   │       │   └── unary.rs            # Placeholder for future SIMD activation optimizations
│   │       └── metal/
│   │           └── mod.rs              # MetalBackend stub (behind feature="metal", not implemented)
│   │
│   ├── ir-model/                       # Model loading + architectures
│   │   └── src/
│   │       ├── lib.rs                  # Re-exports: ModelArchitecture, ModelError
│   │       ├── error.rs                # ModelError enum (Io, InvalidMagic, MissingKey, TensorNotFound, ...)
│   │       ├── architecture.rs         # ModelArchitecture trait: forward(), vocab_size(), reset_cache()
│   │       ├── gguf/
│   │       │   ├── mod.rs              # Re-exports GGUF types
│   │       │   ├── header.rs           # GgufHeader: parse magic (0x47475546), version (v3 only), counts
│   │       │   ├── metadata.rs         # GgufMetadata: parse KV pairs (13 value types), typed getters
│   │       │   ├── tensor_info.rs      # GgufTensorInfo: name/dims/dtype/offset, data_size() calculation
│   │       │   └── reader.rs           # GgufFile: open+parse+mmap, tensor_data(), get_tensor_f32() with
│   │       │                           #   dequantization (F32/F16/Q4_0/Q8_0 -> f32)
│   │       ├── tokenizer/
│   │       │   ├── mod.rs              # Re-exports Vocab, BpeTokenizer
│   │       │   ├── vocab.rs            # Vocab: tokens/scores/bos_id/eos_id/token_to_id from GGUF metadata
│   │       │   └── bpe.rs              # BpeTokenizer: merge_ranks, encode (byte-level BPE), decode
│   │       └── llama/
│   │           ├── mod.rs              # LlamaModel: from_gguf(), ModelArchitecture impl (full forward pass)
│   │           ├── config.rs           # LlamaConfig: n_embd/n_heads/n_kv_heads/n_layers/n_ff/... from GGUF
│   │           ├── layers.rs           # LlamaWeights + LlamaLayer: all weight tensors as flat Vec<f32>
│   │           └── kv_cache.rs         # KvCache: per-layer k/v flat arrays, update/get_k/get_v/reset
│   │
│   ├── ir-sampler/                     # Token sampling strategies
│   │   └── src/
│   │       ├── lib.rs                  # Re-exports all samplers
│   │       ├── sampler.rs              # Sampler trait (apply/reset), SamplerChain (compose + sample)
│   │       ├── temperature.rs          # TemperatureSampler: logit /= temperature
│   │       ├── top_k.rs               # TopKSampler: sort desc, keep top k
│   │       ├── top_p.rs               # TopPSampler: sort desc, softmax, keep until cumulative > p
│   │       ├── repetition.rs           # RepetitionPenaltySampler: penalize recently seen tokens
│   │       └── greedy.rs              # GreedySampler (argmax) + DistSampler (weighted random with seed)
│   │
│   └── ir-ffi/                         # C FFI boundary layer
│       ├── Cargo.toml                  # crate-type = ["cdylib", "staticlib"]
│       ├── cbindgen.toml               # style="type", prefix_with_name=true (critical for CGo)
│       ├── build.rs                    # Runs cbindgen -> go-wrapper/bindings/ir_runtime.h
│       └── src/
│           ├── lib.rs                  # All extern "C" functions: ir_context_create/destroy, ir_model_load,
│           │                           #   ir_generate, ir_generate_streaming, ir_reset, ir_last_error,
│           │                           #   ir_free_string. Each wrapped in catch_panic().
│           ├── types.rs                # #[repr(C)] types: IRStatus, IRBackendType, IRGenerateParams,
│           │                           #   IRStreamCallback
│           ├── context.rs              # IRContext: owns Arc<CpuBackend> + Option<LlamaModel> +
│           │                           #   Option<BpeTokenizer>
│           ├── error.rs                # Thread-local error: set_last_error() / take_last_error()
│           └── streaming.rs            # invoke_callback(): calls IRStreamCallback with CString token
│
├── go-wrapper/                         # Go CLI + HTTP API server
│   ├── go.mod                          # module github.com/cloudchase/inference-runtime, go 1.19
│   ├── main.go                         # Entrypoint: calls cmd.Execute()
│   ├── bindings/
│   │   ├── ir_runtime.h                # Auto-generated C header (cbindgen output, do not edit)
│   │   ├── bridge.h                    # Declares get_go_stream_callback()
│   │   ├── bridge.c                    # C bridge: includes _cgo_export.h, wraps goStreamCallback
│   │   └── runtime.go                  # CGo bindings: Context, NewContext/Close/LoadModel/Generate/
│   │                                   #   GenerateStreaming/Reset, goStreamCallback (//export)
│   ├── engine/
│   │   ├── engine.go                   # Engine: wraps bindings.Context, LoadModel/Generate/GenerateStream/
│   │   │                               #   Reset/Close, model-loaded state tracking
│   │   └── options.go                  # GenerateOptions: Go-native types (float64/int), DefaultOptions()
│   ├── cmd/
│   │   ├── root.go                     # Cobra root command "ir", registers subcommands
│   │   ├── run.go                      # "ir run <model> [prompt]": single-shot or interactive REPL
│   │   ├── serve.go                    # "ir serve [--addr :11434]": starts HTTP API server
│   │   ├── list.go                     # "ir list": tabwriter table of registered models
│   │   └── info.go                     # "ir info <model>": prints model metadata
│   ├── api/
│   │   ├── server.go                   # Server: Engine + ModelManager + Mutex, Start() on http.ServeMux
│   │   ├── routes.go                   # RegisterRoutes: POST /api/generate, POST /api/chat,
│   │   │                               #   GET /api/tags, DELETE /api/delete, GET /api/health
│   │   ├── types.go                    # Request/Response structs: GenerateRequest/Response,
│   │   │                               #   ChatRequest/Response/Message, ListResponse, ErrorResponse
│   │   └── handlers.go                 # Handler implementations: ensureModel (lazy load + mutex),
│   │                                   #   streaming via NDJSON + http.Flusher, chat prompt formatting
│   └── registry/
│       ├── manager.go                  # ModelManager: AddLocalModel, GetModel, ListModels, RemoveModel,
│       │                               #   ResolveModelPath (file path first, then registry lookup)
│       ├── store.go                    # Store: JSON manifest CRUD at ~/.inference-runtime/models/manifests/
│       ├── manifest.go                 # ModelManifest: name, path, size, architecture, quantization, added_at
│       └── download.go                 # Pull(): stub, not implemented
│
└── models/                             # Default model storage directory (GGUF files, gitignored)
```

## Architecture

Two layers connected by C FFI:

```
Go wrapper (cmd/, api/, engine/, registry/) --CGo--> ir-ffi (cdylib) --> ir-model + ir-sampler --> ir-tensor
```

### Rust workspace (`crates/`)

Four crates. Dependency order: `ir-tensor` (leaf) -> `ir-model`, `ir-sampler` -> `ir-ffi` (root).

**ir-tensor** — Tensor library and compute backends. `Tensor` struct holds `CpuStorage` (F32 `Vec<f32>` only in phase 1) + `Shape` + `DType`. The `ComputeBackend` trait (`dyn ComputeBackend`, object-safe, `Send + Sync`) defines all compute ops: `matmul`, `add`, `mul`, `scale`, `rms_norm`, `softmax`, `rope`, `silu`. `CpuBackend` is the sole implementation. Metal backend is a placeholder behind `feature = "metal"` (Cargo.toml optional deps: objc2/objc2-metal). `DType` enum has F32/F16/Q4_0/Q8_0 with GGUF type ID conversion methods. All ops work on `&[f32]` slices and return `Vec<f32>`.

**ir-model** — GGUF parsing, tokenization, and model architectures.
- `gguf/`: Binary parser for GGUF v3 files. `GgufHeader` reads magic/version/counts. `GgufMetadata` parses all 13 GGUF value types into a `HashMap<String, GgufMetadataValue>` with typed getters (`get_u32`, `get_string`, `get_string_array`, etc.). `GgufTensorInfo` stores name/dims/dtype/offset per tensor. `GgufFile` ties it together: opens file, parses header+metadata+tensor_info sequentially via `BufReader`, then memory-maps the whole file with `memmap2::Mmap`. `data_offset` is aligned to 32 bytes. `get_tensor_f32()` dequantizes any format (F32/F16/Q4_0/Q8_0) to `Tensor` with f32 data. Q4_0: 18-byte blocks (f16 scale + 16 bytes of packed nibbles, value = (nibble - 8) * scale). Q8_0: 34-byte blocks (f16 scale + 32 signed bytes, value = byte * scale).
- `tokenizer/`: `Vocab` loads token strings, scores, bos/eos IDs, and reverse map from GGUF metadata keys `tokenizer.ggml.*`. `BpeTokenizer` loads merge rules from `tokenizer.ggml.merges`, stores `merge_ranks: HashMap<(String, String), usize>`. Encode: bytes -> per-byte vocab lookup (tries char then `<0xHH>` format) -> iterative lowest-rank merge. Decode: token IDs -> strings (byte tokens `<0xHH>` converted back to bytes).
- `llama/`: `LlamaConfig` parsed from GGUF metadata keys (`llama.embedding_length`, `llama.attention.head_count`, etc.). `LlamaWeights` loads all named tensors (`token_embd.weight`, `blk.{i}.attn_q.weight`, etc.) as flat `Vec<f32>`; supports tied embeddings (output falls back to token_embd). `LlamaLayer` holds per-layer weights (attn_norm, wq/wk/wv/wo, ffn_norm, ffn_gate/ffn_up/ffn_down). `KvCache`: flat `Vec<Vec<f32>>` per layer, layout `[max_seq_len, n_kv_heads * head_dim]`. Forward pass processes one token at a time through all layers: embedding lookup -> per-layer (RMS norm -> QKV projection as matmul(W, x, out_dim, in_dim, 1) -> RoPE -> KV cache update -> GQA attention with inline softmax -> output projection -> residual -> RMS norm -> SwiGLU FFN -> residual) -> final norm -> logits projection. GQA: `heads_per_kv = n_heads / n_kv_heads`, each query head shares K/V from head `h / heads_per_kv`. Causal masking is implicit (cache only has positions <= current). Only the last token's logits are computed.
- `ModelArchitecture` trait: `forward(&mut self, tokens, pos, backend) -> Vec<f32>`, `vocab_size()`, `reset_cache()`.

**ir-sampler** — Token sampling. `Sampler` trait: `apply(&self, &mut Vec<TokenLogit>)` modifies logits in place. `SamplerChain` composes samplers sequentially, returns `token_logits[0].token_id`. Implementations: `TemperatureSampler` (divides logits by temp), `TopKSampler` (sort desc, truncate to k), `TopPSampler` (sort desc, compute softmax probs, truncate at cumulative > p), `RepetitionPenaltySampler` (divides positive logits / multiplies negative logits by penalty for recent tokens), `GreedySampler` (sort desc, truncate to 1), `DistSampler` (softmax -> weighted random sample with seeded RNG). No dependency on ir-tensor (uses `rand` crate only).

**ir-ffi** — C API boundary. Built as both `cdylib` and `staticlib`. `build.rs` runs cbindgen to generate `go-wrapper/bindings/ir_runtime.h`. Types: `IRStatus` (repr(C) enum: Ok=0 through ErrorInternal=5), `IRBackendType` (Cpu=0, Metal=1), `IRGenerateParams` (repr(C) struct), `IRStreamCallback` (Option<extern "C" fn>). `IRContext` (context.rs) owns `Arc<CpuBackend>`, `Option<LlamaModel>`, `Option<BpeTokenizer>`. Error handling: thread-local `RefCell<Option<CString>>` via `set_last_error`/`take_last_error`. All public FFI functions wrapped in `catch_panic` which converts panics to `IRStatus::ErrorInternal`. `ir_generate` builds a `SamplerChain` (RepetitionPenalty -> Temperature -> TopK -> TopP -> Greedy), runs prefill on all prompt tokens, then decodes one token at a time checking for EOS. `ir_generate_streaming` is similar but calls `streaming::invoke_callback` per token. `ir_free_string` reclaims `CString`s.

### cbindgen configuration (`crates/ir-ffi/cbindgen.toml`)

`style = "type"` is required — `"both"` generates named enums that CGo cannot cast correctly. `prefix_with_name = true` under `[enum]` produces `IR_STATUS_OK`, `IR_BACKEND_TYPE_CPU` etc. that the Go bindings reference.

### Go wrapper (`go-wrapper/`)

**bindings/** — `runtime.go`: CGo preamble links `libir_ffi` and includes `ir_runtime.h` + `bridge.h`. Go types mirror C types. `NewContext` / `Close` / `LoadModel` / `Generate` / `GenerateStreaming` / `Reset` wrap the C functions. Streaming uses `cgo.Handle` to pass a Go `*streamHandle` through C `void*`. `bridge.c`/`bridge.h`: C bridge pattern — bridge.c includes `_cgo_export.h` (auto-generated by CGo at compile time, unavailable in the CGo comment block) and defines `streamBridge` which calls the Go `goStreamCallback`, then `get_go_stream_callback()` returns it as an `IRStreamCallback`. This avoids the `bool`/`_Bool` type conflict from declaring the Go callback as `extern` in the CGo preamble.

**engine/** — `Engine` struct wraps `bindings.Context` with model-loaded state tracking. `GenerateOptions` uses Go-native `float64`/`int` types, converted to `bindings.GenerateParams` (`float32`/`uint32`) at call sites.

**cmd/** — Cobra CLI. Root command `ir`. Subcommands: `run` (single-shot or interactive REPL with `/exit`, `/reset`, `/help`), `serve` (starts HTTP server, default `:11434`), `list` (tabwriter output of registered models), `info` (prints model metadata). Model resolution: tries filesystem path first, then registry lookup.

**api/** — `Server` struct holds `*engine.Engine`, `*registry.ModelManager`, `sync.Mutex` for model loading. Routes (registered with Go 1.22-style method patterns on `http.ServeMux`): `POST /api/generate`, `POST /api/chat`, `GET /api/tags`, `DELETE /api/delete`, `GET /api/health`. Streaming uses NDJSON (newline-delimited JSON) with `http.Flusher`. Chat endpoint converts messages to a flat prompt (`"System: ... User: ... Assistant: "`). `ensureModel` lazy-loads models, resets if switching.

**registry/** — `ModelManager` -> `Store`. Manifests stored as JSON at `~/.inference-runtime/models/manifests/{name}.json`. `ModelManifest`: name, path, size, architecture, parameters, quantization, added_at. `ResolveModelPath`: checks if arg is an existing file path first, then looks up registry. `Pull` is a stub (not implemented).

## Key Patterns and Gotchas

- All weights stored as flat `Vec<f32>`, not `Tensor`. Weight matrices use row-major `[out_dim, in_dim]` layout. Matrix-vector product for single-token inference: `backend.matmul(weight, input, out_dim, in_dim, 1)`.
- `CpuStorage` only has an `F32` variant. The `DType` enum and quantized types exist for GGUF parsing and dequantization but all compute runs on f32.
- The sampler chain in `ir_generate` differs from `ir_generate_streaming`: the non-streaming version includes `RepetitionPenaltySampler`, the streaming version does not.
- `ir_last_error` returns a string the caller must free with `ir_free_string`. The Go `LastError()` function reads the C string but does not free it (the C string is from a thread-local that was `take`n, so it's already consumed).
- Rust release profile: `lto = true`, `codegen-units = 1`, `opt-level = 3`.
