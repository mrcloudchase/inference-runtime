package registry

import "fmt"

// Pull downloads a model from a remote registry (e.g. HuggingFace).
// This is not yet implemented; use local GGUF files directly for now.
func (m *ModelManager) Pull(name string) error {
	// TODO: Implement HuggingFace model downloading
	return fmt.Errorf("pull not yet implemented: use 'ir run /path/to/model.gguf' for local files")
}
