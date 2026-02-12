package api

// GenerateRequest is the JSON body for POST /api/generate.
type GenerateRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

// GenerateResponse is the JSON response for POST /api/generate.
type GenerateResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// ChatMessage represents a single message in a chat conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the JSON body for POST /api/chat.
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream,omitempty"`
}

// ChatResponse is the JSON response for POST /api/chat.
type ChatResponse struct {
	Model   string      `json:"model"`
	Message ChatMessage `json:"message"`
	Done    bool        `json:"done"`
}

// ModelInfo describes a model in list responses.
type ModelInfo struct {
	Name         string `json:"name"`
	Size         int64  `json:"size"`
	Architecture string `json:"architecture,omitempty"`
	Quantization string `json:"quantization,omitempty"`
}

// ListResponse is the JSON response for GET /api/tags.
type ListResponse struct {
	Models []ModelInfo `json:"models"`
}

// ErrorResponse is the JSON error envelope.
type ErrorResponse struct {
	Error string `json:"error"`
}

// DeleteRequest is the JSON body for DELETE /api/delete.
type DeleteRequest struct {
	Name string `json:"name"`
}
