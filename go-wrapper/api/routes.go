package api

import "net/http"

// RegisterRoutes wires up the API endpoints on the given ServeMux.
func RegisterRoutes(mux *http.ServeMux, s *Server) {
	mux.HandleFunc("POST /api/generate", s.handleGenerate)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("GET /api/tags", s.handleListModels)
	mux.HandleFunc("DELETE /api/delete", s.handleDeleteModel)
	mux.HandleFunc("GET /api/health", s.handleHealth)
}
