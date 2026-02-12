package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/cloudchase/inference-runtime/engine"
)

// writeJSON writes a JSON response with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("failed to write JSON response: %v", err)
	}
}

// writeError writes an error response with the given status code.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, ErrorResponse{Error: msg})
}

// ensureModel resolves the model path and loads it if not already loaded.
func (s *Server) ensureModel(model string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.engine.IsLoaded() && s.engine.ModelPath() == model {
		return nil
	}

	modelPath, err := s.manager.ResolveModelPath(model)
	if err != nil {
		return fmt.Errorf("cannot resolve model: %w", err)
	}

	// If a different model is already loaded, reset first.
	if s.engine.IsLoaded() {
		if err := s.engine.Reset(); err != nil {
			return fmt.Errorf("reset failed: %w", err)
		}
	}

	if err := s.engine.LoadModel(modelPath); err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	return nil
}

// buildOptions converts an API generate request into engine options.
func buildOptions(req GenerateRequest) engine.GenerateOptions {
	opts := engine.DefaultOptions()
	if req.MaxTokens > 0 {
		opts.MaxTokens = req.MaxTokens
	}
	if req.Temperature > 0 {
		opts.Temperature = req.Temperature
	}
	if req.TopK > 0 {
		opts.TopK = req.TopK
	}
	if req.TopP > 0 {
		opts.TopP = req.TopP
	}
	opts.Stream = req.Stream
	return opts
}

// handleGenerate handles POST /api/generate.
// Supports both streaming (SSE with JSON lines) and non-streaming responses.
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	opts := buildOptions(req)

	if req.Stream {
		s.handleGenerateStream(w, r, req, opts)
		return
	}

	output, err := s.engine.Generate(req.Prompt, opts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "generation failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, GenerateResponse{
		Model:    req.Model,
		Response: output,
		Done:     true,
	})
}

// handleGenerateStream handles streaming generation via SSE / JSON lines.
func (s *Server) handleGenerateStream(w http.ResponseWriter, _ *http.Request, req GenerateRequest, opts engine.GenerateOptions) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	encoder := json.NewEncoder(w)

	err := s.engine.GenerateStream(req.Prompt, opts, func(token string) bool {
		resp := GenerateResponse{
			Model:    req.Model,
			Response: token,
			Done:     false,
		}
		if encErr := encoder.Encode(resp); encErr != nil {
			log.Printf("stream encode error: %v", encErr)
			return false
		}
		flusher.Flush()
		return true
	})

	if err != nil {
		log.Printf("streaming generation error: %v", err)
	}

	// Send final done message.
	_ = encoder.Encode(GenerateResponse{
		Model:    req.Model,
		Response: "",
		Done:     true,
	})
	flusher.Flush()
}

// handleChat handles POST /api/chat.
// Converts chat messages to a prompt and delegates to generation.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages are required")
		return
	}

	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Build a simple prompt from messages.
	var sb strings.Builder
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			sb.WriteString("System: ")
		case "user":
			sb.WriteString("User: ")
		case "assistant":
			sb.WriteString("Assistant: ")
		default:
			sb.WriteString(msg.Role + ": ")
		}
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
	}
	sb.WriteString("Assistant: ")
	prompt := sb.String()

	opts := engine.DefaultOptions()
	opts.Stream = req.Stream

	if req.Stream {
		s.handleChatStream(w, r, req, prompt, opts)
		return
	}

	output, err := s.engine.Generate(prompt, opts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "generation failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, ChatResponse{
		Model: req.Model,
		Message: ChatMessage{
			Role:    "assistant",
			Content: output,
		},
		Done: true,
	})
}

// handleChatStream handles streaming chat generation.
func (s *Server) handleChatStream(w http.ResponseWriter, _ *http.Request, req ChatRequest, prompt string, opts engine.GenerateOptions) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	encoder := json.NewEncoder(w)

	err := s.engine.GenerateStream(prompt, opts, func(token string) bool {
		resp := ChatResponse{
			Model: req.Model,
			Message: ChatMessage{
				Role:    "assistant",
				Content: token,
			},
			Done: false,
		}
		if encErr := encoder.Encode(resp); encErr != nil {
			log.Printf("chat stream encode error: %v", encErr)
			return false
		}
		flusher.Flush()
		return true
	})

	if err != nil {
		log.Printf("streaming chat error: %v", err)
	}

	// Send final done message.
	_ = encoder.Encode(ChatResponse{
		Model: req.Model,
		Message: ChatMessage{
			Role:    "assistant",
			Content: "",
		},
		Done: true,
	})
	flusher.Flush()
}

// handleListModels handles GET /api/tags.
func (s *Server) handleListModels(w http.ResponseWriter, _ *http.Request) {
	manifests, err := s.manager.ListModels()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list models: "+err.Error())
		return
	}

	models := make([]ModelInfo, 0, len(manifests))
	for _, m := range manifests {
		models = append(models, ModelInfo{
			Name:         m.Name,
			Size:         m.Size,
			Architecture: m.Architecture,
			Quantization: m.Quantization,
		})
	}

	writeJSON(w, http.StatusOK, ListResponse{Models: models})
}

// handleDeleteModel handles DELETE /api/delete.
func (s *Server) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	var req DeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required")
		return
	}

	if err := s.manager.RemoveModel(req.Name); err != nil {
		writeError(w, http.StatusNotFound, "model not found: "+err.Error())
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// handleHealth handles GET /api/health.
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	resp := map[string]any{
		"status":       "ok",
		"model_loaded": s.engine.IsLoaded(),
	}
	if s.engine.IsLoaded() {
		resp["model"] = s.engine.ModelPath()
	}
	writeJSON(w, http.StatusOK, resp)
}
