package engine

// GenerateOptions holds generation parameters at the engine level,
// using Go-native types (float64/int) for ergonomic API usage.
type GenerateOptions struct {
	MaxTokens        int
	Temperature      float64
	TopK             int
	TopP             float64
	RepetitionPenalty float64
	Seed             uint64
	Stream           bool
}

// DefaultOptions returns sensible generation defaults.
func DefaultOptions() GenerateOptions {
	return GenerateOptions{
		MaxTokens:        256,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.95,
		RepetitionPenalty: 1.1,
	}
}
