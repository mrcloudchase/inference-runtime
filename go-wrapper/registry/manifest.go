package registry

import "time"

// ModelManifest describes a registered model's metadata.
type ModelManifest struct {
	Name         string    `json:"name"`
	Path         string    `json:"path"`
	Size         int64     `json:"size"`
	Architecture string    `json:"architecture,omitempty"`
	Parameters   string    `json:"parameters,omitempty"`
	Quantization string    `json:"quantization,omitempty"`
	AddedAt      time.Time `json:"added_at"`
}
