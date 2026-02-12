package api

import (
	"log"
	"net/http"
	"sync"

	"github.com/cloudchase/inference-runtime/engine"
	"github.com/cloudchase/inference-runtime/registry"
)

// Server is the HTTP API server for the inference runtime.
type Server struct {
	engine  *engine.Engine
	manager *registry.ModelManager
	addr    string
	mu      sync.Mutex // guards model loading
}

// NewServer creates a new API server.
func NewServer(eng *engine.Engine, mgr *registry.ModelManager, addr string) *Server {
	return &Server{
		engine:  eng,
		manager: mgr,
		addr:    addr,
	}
}

// Start registers routes and starts the HTTP server (blocking).
func (s *Server) Start() error {
	mux := http.NewServeMux()
	RegisterRoutes(mux, s)

	log.Printf("Starting inference-runtime API server on %s", s.addr)
	return http.ListenAndServe(s.addr, mux)
}
