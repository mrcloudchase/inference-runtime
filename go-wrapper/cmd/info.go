package cmd

import (
	"fmt"

	"github.com/cloudchase/inference-runtime/registry"
	"github.com/spf13/cobra"
)

var infoCmd = &cobra.Command{
	Use:   "info <model>",
	Short: "Show model information",
	Long:  "Display detailed metadata for a registered model.",
	Args:  cobra.ExactArgs(1),
	RunE:  runInfo,
}

func runInfo(_ *cobra.Command, args []string) error {
	name := args[0]

	mgr, err := registry.NewModelManager(registry.DefaultBaseDir())
	if err != nil {
		return fmt.Errorf("init model manager: %w", err)
	}

	m, err := mgr.GetModel(name)
	if err != nil {
		return fmt.Errorf("model '%s' not found: %w", name, err)
	}

	fmt.Printf("Name:          %s\n", m.Name)
	fmt.Printf("Path:          %s\n", m.Path)
	fmt.Printf("Size:          %s\n", formatSize(m.Size))
	if m.Architecture != "" {
		fmt.Printf("Architecture:  %s\n", m.Architecture)
	}
	if m.Parameters != "" {
		fmt.Printf("Parameters:    %s\n", m.Parameters)
	}
	if m.Quantization != "" {
		fmt.Printf("Quantization:  %s\n", m.Quantization)
	}
	fmt.Printf("Added:         %s\n", m.AddedAt.Format("2006-01-02 15:04:05"))

	return nil
}
