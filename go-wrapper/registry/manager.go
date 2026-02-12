package registry

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// ModelManager provides high-level operations for managing local models.
type ModelManager struct {
	store *Store
}

// NewModelManager creates a ModelManager and ensures the storage directories exist.
func NewModelManager(baseDir string) (*ModelManager, error) {
	store := NewStore(baseDir)
	if err := store.EnsureDirs(); err != nil {
		return nil, err
	}
	return &ModelManager{store: store}, nil
}

// DefaultBaseDir returns the default base directory (~/.inference-runtime).
func DefaultBaseDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".inference-runtime")
}

// AddLocalModel registers an existing GGUF file as a named model.
func (m *ModelManager) AddLocalModel(name, ggufPath string) error {
	abs, err := filepath.Abs(ggufPath)
	if err != nil {
		return err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}
	manifest := &ModelManifest{
		Name:    name,
		Path:    abs,
		Size:    info.Size(),
		AddedAt: time.Now(),
	}
	return m.store.SaveManifest(manifest)
}

// GetModel retrieves a model manifest by name.
func (m *ModelManager) GetModel(name string) (*ModelManifest, error) {
	return m.store.LoadManifest(name)
}

// ListModels returns all registered model manifests.
func (m *ModelManager) ListModels() ([]ModelManifest, error) {
	return m.store.ListManifests()
}

// RemoveModel deletes a model's manifest from the registry.
func (m *ModelManager) RemoveModel(name string) error {
	return m.store.DeleteManifest(name)
}

// ResolveModelPath resolves a model name or file path to an absolute file path.
// If nameOrPath is an existing file, it is returned directly.
// Otherwise, the registry is consulted for a matching model name.
func (m *ModelManager) ResolveModelPath(nameOrPath string) (string, error) {
	if _, err := os.Stat(nameOrPath); err == nil {
		abs, err := filepath.Abs(nameOrPath)
		if err != nil {
			return nameOrPath, nil
		}
		return abs, nil
	}
	manifest, err := m.store.LoadManifest(nameOrPath)
	if err != nil {
		return "", fmt.Errorf("model '%s' not found locally and not a valid path", nameOrPath)
	}
	return manifest.Path, nil
}
