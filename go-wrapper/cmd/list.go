package cmd

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/cloudchase/inference-runtime/registry"
	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
	Use:     "list",
	Aliases: []string{"ls"},
	Short:   "List local models",
	Long:    "List all models registered in the local model registry.",
	RunE:    runList,
}

func runList(_ *cobra.Command, _ []string) error {
	mgr, err := registry.NewModelManager(registry.DefaultBaseDir())
	if err != nil {
		return fmt.Errorf("init model manager: %w", err)
	}

	models, err := mgr.ListModels()
	if err != nil {
		return fmt.Errorf("list models: %w", err)
	}

	if len(models) == 0 {
		fmt.Println("No models registered. Use 'ir run /path/to/model.gguf' to run a local model.")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "NAME\tSIZE\tQUANTIZATION\tARCHITECTURE\tADDED")
	for _, m := range models {
		size := formatSize(m.Size)
		quant := m.Quantization
		if quant == "" {
			quant = "-"
		}
		arch := m.Architecture
		if arch == "" {
			arch = "-"
		}
		added := m.AddedAt.Format("2006-01-02 15:04")
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", m.Name, size, quant, arch, added)
	}
	return w.Flush()
}

func formatSize(bytes int64) string {
	const (
		kb = 1024
		mb = kb * 1024
		gb = mb * 1024
	)
	switch {
	case bytes >= gb:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	case bytes >= mb:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(mb))
	case bytes >= kb:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(kb))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}
