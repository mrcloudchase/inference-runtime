# inference-runtime

[![Stars](https://img.shields.io/github/stars/mrcloudchase/inference-runtime)](https://github.com/mrcloudchase/inference-runtime/stargazers)
[![Forks](https://img.shields.io/github/forks/mrcloudchase/inference-runtime)](https://github.com/mrcloudchase/inference-runtime/network/members)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mrcloudchase/inference-runtime/blob/main/LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/mrcloudchase/inference-runtime)](https://github.com/mrcloudchase/inference-runtime/commits/main)
[![Top language](https://img.shields.io/github/languages/top/mrcloudchase/inference-runtime)](https://github.com/mrcloudchase/inference-runtime)

A custom LLM inference runtime built from scratch in Rust and Go. The Rust engine handles tensor operations, GGUF model loading, and the transformer forward pass. The Go wrapper provides a CLI and REST API.

```
┌─────────────────────────────────────────────────┐
│  Go Wrapper (CLI + REST API + Model Management) │
│    cmd/  api/  registry/  engine/               │
└──────────────────┬──────────────────────────────┘
                   │ CGo FFI
┌──────────────────▼──────────────────────────────┐
│  ir-ffi          │ C API boundary (cdylib)       │
├──────────────────┼──────────────────────────────┤
│  ir-model        │ GGUF parser, LLaMA forward    │
│                  │ pass, BPE tokenizer, KV cache  │
├──────────────────┼──────────────────────────────┤
│  ir-sampler      │ Temperature, Top-K/P,          │
│                  │ repetition penalty              │
├──────────────────┼──────────────────────────────┤
│  ir-tensor       │ Tensor ops, ComputeBackend     │
│                  │ trait, CPU backend              │
└─────────────────────────────────────────────────┘
```

## Requirements

- Rust 1.70+ (with `cargo`)
- Go 1.22+
- macOS / Apple Silicon (arm64)

## Build

```sh
make          # builds Rust crates, then Go binary
```

The final binary is at `target/release/ir`.

To build each layer independently:

```sh
make rust     # cargo build --release
make go       # go build with CGo linking
```

## Usage

### Run a model

```sh
# Single-shot generation
ir run /path/to/model.gguf "Once upon a time"

# Interactive REPL
ir run /path/to/model.gguf
>>> Tell me a joke
>>> /reset
>>> /exit
```

### Start the API server

```sh
ir serve                    # default :11434
ir serve --addr :8080       # custom port
```

### Manage models

```sh
ir list                     # show registered models
ir info <model>             # show model metadata
```

## REST API

The server exposes these endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Text completion (streaming NDJSON) |
| `POST` | `/api/chat` | Chat completion (streaming NDJSON) |
| `GET` | `/api/tags` | List local models |
| `DELETE` | `/api/delete` | Remove a local model |
| `GET` | `/api/health` | Health check |

### Generate text

```sh
curl http://localhost:11434/api/generate -d '{
  "model": "/path/to/model.gguf",
  "prompt": "Hello, world!",
  "stream": true
}'
```

### Chat

```sh
curl http://localhost:11434/api/chat -d '{
  "model": "/path/to/model.gguf",
  "messages": [{"role": "user", "content": "Hi"}],
  "stream": false
}'
```

## Project Structure

```
inference-runtime/
├── Cargo.toml                  # Rust workspace root
├── Makefile
├── crates/
│   ├── ir-tensor/              # Tensor library + CPU backend
│   │   └── src/
│   │       ├── tensor.rs       # Tensor struct + reshape/matmul
│   │       ├── backend.rs      # ComputeBackend trait
│   │       ├── cpu/            # CPU implementations
│   │       ├── dtype.rs        # F32, F16, Q4_0, Q8_0
│   │       └── shape.rs        # Shape + broadcasting
│   │
│   ├── ir-model/               # Model loading + architectures
│   │   └── src/
│   │       ├── gguf/           # GGUF v3 parser (mmap-backed)
│   │       ├── tokenizer/      # BPE tokenizer from GGUF metadata
│   │       ├── llama/          # LLaMA forward pass + KV cache
│   │       └── architecture.rs # ModelArchitecture trait
│   │
│   ├── ir-sampler/             # Sampling strategies
│   │   └── src/
│   │       ├── sampler.rs      # Sampler trait + SamplerChain
│   │       ├── temperature.rs
│   │       ├── top_k.rs
│   │       ├── top_p.rs
│   │       ├── repetition.rs
│   │       └── greedy.rs
│   │
│   └── ir-ffi/                 # C FFI boundary
│       ├── cbindgen.toml       # Generates ir_runtime.h
│       └── src/
│           ├── lib.rs          # extern "C" functions
│           ├── types.rs        # #[repr(C)] structs/enums
│           ├── context.rs      # IRContext opaque handle
│           └── streaming.rs    # Callback-based streaming
│
├── go-wrapper/                 # Go CLI + API server
│   ├── main.go
│   ├── bindings/               # CGo bindings + generated C header
│   ├── cmd/                    # CLI commands (cobra)
│   ├── api/                    # REST API server
│   ├── engine/                 # Go-side inference abstraction
│   └── registry/               # Local model management
│
└── models/                     # Default model storage
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **GGUF format** | Use existing quantized models from HuggingFace directly |
| **mmap for model loading** | Zero-copy, OS handles paging — a 4GB model doesn't need 4GB malloc |
| **`dyn ComputeBackend` trait** | Runtime backend selection; vtable cost is negligible vs tensor op cost |
| **Separate Rust crates** | Enforces dependency boundaries at compile time |
| **Thread-local FFI errors** | Safe for Go's goroutine-to-thread mapping, like C `errno` |
| **cbindgen** | Auto-generated C header, always in sync with Rust types |
| **Cobra for CLI** | Industry standard for Go CLI tools |

## Supported Model Formats

- **GGUF v3** with the following tensor types:
  - F32 (unquantized)
  - F16 (half precision)
  - Q4_0 (4-bit block quantization)
  - Q8_0 (8-bit block quantization)

## Testing

```sh
make test     # runs cargo test + go test
make check    # runs cargo clippy
make fmt      # formats Rust + Go code
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan. Current status:

- **Phase 1** — CPU inference end-to-end (complete)
- **Phase 2** — Metal GPU acceleration
- **Phase 3** — Performance optimization (SIMD, quantized compute)
- **Phase 4** — Additional architectures (Mistral, Phi, Gemma)
- **Phase 5** — Model management (HuggingFace downloads)
- **Phase 6** — Production features (OpenAI-compatible API, CUDA, Docker)

## License

MIT
