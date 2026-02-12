package registry

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Store manages on-disk storage for model manifests and blobs.
type Store struct {
	baseDir string
}

// NewStore creates a Store rooted at baseDir.
func NewStore(baseDir string) *Store {
	return &Store{baseDir: baseDir}
}

// ModelsDir returns the top-level models directory.
func (s *Store) ModelsDir() string { return filepath.Join(s.baseDir, "models") }

// ManifestsDir returns the directory where model manifests are stored.
func (s *Store) ManifestsDir() string { return filepath.Join(s.baseDir, "models", "manifests") }

// BlobsDir returns the directory where model blobs are stored.
func (s *Store) BlobsDir() string { return filepath.Join(s.baseDir, "models", "blobs") }

// EnsureDirs creates the required directory structure if it does not exist.
func (s *Store) EnsureDirs() error {
	for _, d := range []string{s.ManifestsDir(), s.BlobsDir()} {
		if err := os.MkdirAll(d, 0755); err != nil {
			return err
		}
	}
	return nil
}

// SaveManifest writes a model manifest to disk as JSON.
func (s *Store) SaveManifest(m *ModelManifest) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(s.ManifestsDir(), m.Name+".json")
	return os.WriteFile(path, data, 0644)
}

// LoadManifest reads a model manifest from disk by name.
func (s *Store) LoadManifest(name string) (*ModelManifest, error) {
	path := filepath.Join(s.ManifestsDir(), name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m ModelManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// ListManifests returns all model manifests found in the manifests directory.
func (s *Store) ListManifests() ([]ModelManifest, error) {
	entries, err := os.ReadDir(s.ManifestsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var manifests []ModelManifest
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".json" {
			continue
		}
		name := e.Name()[:len(e.Name())-5]
		m, err := s.LoadManifest(name)
		if err != nil {
			continue
		}
		manifests = append(manifests, *m)
	}
	return manifests, nil
}

// DeleteManifest removes a model manifest from disk.
func (s *Store) DeleteManifest(name string) error {
	path := filepath.Join(s.ManifestsDir(), name+".json")
	return os.Remove(path)
}
