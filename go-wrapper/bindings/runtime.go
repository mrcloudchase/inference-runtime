package bindings

/*
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lir_ffi
#cgo CFLAGS: -I${SRCDIR}
#include "ir_runtime.h"
#include "bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime/cgo"
	"unsafe"
)

// Status represents the return status from the FFI layer.
type Status int

const (
	StatusOK              Status = C.IR_STATUS_OK
	StatusInvalidArgument Status = C.IR_STATUS_ERROR_INVALID_ARGUMENT
	StatusModelLoadError  Status = C.IR_STATUS_ERROR_MODEL_LOAD
	StatusGenerateError   Status = C.IR_STATUS_ERROR_GENERATE
	StatusOutOfMemory     Status = C.IR_STATUS_ERROR_OUT_OF_MEMORY
	StatusInternalError   Status = C.IR_STATUS_ERROR_INTERNAL
)

// BackendType selects the compute backend for inference.
type BackendType int

const (
	BackendCPU   BackendType = C.IR_BACKEND_TYPE_CPU
	BackendMetal BackendType = C.IR_BACKEND_TYPE_METAL
)

// GenerateParams mirrors the C IRGenerateParams struct.
type GenerateParams struct {
	MaxTokens        uint32
	Temperature      float32
	TopK             uint32
	TopP             float32
	RepetitionPenalty float32
	Seed             uint64
}

// DefaultGenerateParams returns sensible defaults for generation.
func DefaultGenerateParams() GenerateParams {
	return GenerateParams{
		MaxTokens:        256,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.95,
		RepetitionPenalty: 1.1,
		Seed:             0,
	}
}

// Context wraps an opaque IRContext pointer from the FFI layer.
type Context struct {
	ctx *C.IRContext
}

// NewContext creates a new inference context with the specified backend.
func NewContext(backend BackendType) (*Context, error) {
	var ctx *C.IRContext
	status := C.ir_context_create(C.IRBackendType(backend), &ctx)
	if status != C.IR_STATUS_OK {
		return nil, fmt.Errorf("failed to create context: %s", LastError())
	}
	return &Context{ctx: ctx}, nil
}

// Close destroys the underlying context and frees resources.
func (c *Context) Close() {
	if c.ctx != nil {
		C.ir_context_destroy(c.ctx)
		c.ctx = nil
	}
}

// LoadModel loads a GGUF model file into the context.
func (c *Context) LoadModel(path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	status := C.ir_model_load(c.ctx, cPath)
	if status != C.IR_STATUS_OK {
		return fmt.Errorf("failed to load model: %s", LastError())
	}
	return nil
}

// Generate runs non-streaming generation and returns the full output.
func (c *Context) Generate(prompt string, params GenerateParams) (string, error) {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cParams := C.IRGenerateParams{
		max_tokens:        C.uint32_t(params.MaxTokens),
		temperature:       C.float(params.Temperature),
		top_k:             C.uint32_t(params.TopK),
		top_p:             C.float(params.TopP),
		repetition_penalty: C.float(params.RepetitionPenalty),
		seed:              C.uint64_t(params.Seed),
	}

	var output *C.char
	status := C.ir_generate(c.ctx, cPrompt, cParams, &output)
	if status != C.IR_STATUS_OK {
		return "", fmt.Errorf("generation failed: %s", LastError())
	}

	result := C.GoString(output)
	C.ir_free_string(output)
	return result, nil
}

// StreamCallback is called for each generated token. Return false to stop generation.
type StreamCallback func(token string) bool

type streamHandle struct {
	callback StreamCallback
}

//export goStreamCallback
func goStreamCallback(token *C.char, userData unsafe.Pointer) C.bool {
	h := cgo.Handle(uintptr(userData))
	sh := h.Value().(*streamHandle)
	goToken := C.GoString(token)
	if sh.callback(goToken) {
		return C.bool(true)
	}
	return C.bool(false)
}

// GenerateStreaming runs streaming generation, calling callback for each token.
func (c *Context) GenerateStreaming(prompt string, params GenerateParams, callback StreamCallback) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cParams := C.IRGenerateParams{
		max_tokens:        C.uint32_t(params.MaxTokens),
		temperature:       C.float(params.Temperature),
		top_k:             C.uint32_t(params.TopK),
		top_p:             C.float(params.TopP),
		repetition_penalty: C.float(params.RepetitionPenalty),
		seed:              C.uint64_t(params.Seed),
	}

	sh := &streamHandle{callback: callback}
	h := cgo.NewHandle(sh)
	defer h.Delete()

	status := C.ir_generate_streaming(
		c.ctx,
		cPrompt,
		cParams,
		C.get_go_stream_callback(),
		unsafe.Pointer(uintptr(h)), //nolint:govet // cgo.Handle is a uintptr; this round-trip is safe
	)
	if status != C.IR_STATUS_OK {
		return fmt.Errorf("streaming generation failed: %s", LastError())
	}
	return nil
}

// Reset clears the context's KV cache and state.
func (c *Context) Reset() error {
	status := C.ir_reset(c.ctx)
	if status != C.IR_STATUS_OK {
		return fmt.Errorf("reset failed: %s", LastError())
	}
	return nil
}

// LastError returns the most recent error message from the FFI layer.
func LastError() string {
	cErr := C.ir_last_error()
	if cErr == nil {
		return "unknown error"
	}
	return C.GoString(cErr)
}
