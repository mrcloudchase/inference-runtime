package engine

import (
	"fmt"

	"github.com/cloudchase/inference-runtime/bindings"
)

// Engine wraps the low-level bindings.Context and provides a
// higher-level interface for model loading and text generation.
type Engine struct {
	ctx       *bindings.Context
	modelPath string
	loaded    bool
}

// New creates a new Engine with a CPU backend context.
func New() (*Engine, error) {
	ctx, err := bindings.NewContext(bindings.BackendCPU)
	if err != nil {
		return nil, fmt.Errorf("engine init: %w", err)
	}
	return &Engine{ctx: ctx}, nil
}

// NewWithBackend creates a new Engine with the specified backend.
func NewWithBackend(backend bindings.BackendType) (*Engine, error) {
	ctx, err := bindings.NewContext(backend)
	if err != nil {
		return nil, fmt.Errorf("engine init: %w", err)
	}
	return &Engine{ctx: ctx}, nil
}

// LoadModel loads a GGUF model file into the engine.
func (e *Engine) LoadModel(path string) error {
	if err := e.ctx.LoadModel(path); err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	e.modelPath = path
	e.loaded = true
	return nil
}

// Generate runs non-streaming text generation and returns the full output.
func (e *Engine) Generate(prompt string, opts GenerateOptions) (string, error) {
	if !e.loaded {
		return "", fmt.Errorf("no model loaded")
	}

	params := bindings.GenerateParams{
		MaxTokens:        uint32(opts.MaxTokens),
		Temperature:      float32(opts.Temperature),
		TopK:             uint32(opts.TopK),
		TopP:             float32(opts.TopP),
		RepetitionPenalty: float32(opts.RepetitionPenalty),
		Seed:             opts.Seed,
	}

	return e.ctx.Generate(prompt, params)
}

// GenerateStream runs streaming text generation, calling callback for each token.
// Return false from callback to stop generation early.
func (e *Engine) GenerateStream(prompt string, opts GenerateOptions, callback func(string) bool) error {
	if !e.loaded {
		return fmt.Errorf("no model loaded")
	}

	params := bindings.GenerateParams{
		MaxTokens:        uint32(opts.MaxTokens),
		Temperature:      float32(opts.Temperature),
		TopK:             uint32(opts.TopK),
		TopP:             float32(opts.TopP),
		RepetitionPenalty: float32(opts.RepetitionPenalty),
		Seed:             opts.Seed,
	}

	return e.ctx.GenerateStreaming(prompt, params, callback)
}

// Reset clears the engine's KV cache and internal state.
func (e *Engine) Reset() error {
	if e.ctx == nil {
		return nil
	}
	return e.ctx.Reset()
}

// IsLoaded returns whether a model is currently loaded.
func (e *Engine) IsLoaded() bool { return e.loaded }

// ModelPath returns the path of the currently loaded model.
func (e *Engine) ModelPath() string { return e.modelPath }

// Close destroys the underlying context and frees all resources.
func (e *Engine) Close() {
	if e.ctx != nil {
		e.ctx.Close()
		e.ctx = nil
	}
	e.loaded = false
}
