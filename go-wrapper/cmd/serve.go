package cmd

import (
	"fmt"

	"github.com/cloudchase/inference-runtime/api"
	"github.com/cloudchase/inference-runtime/engine"
	"github.com/cloudchase/inference-runtime/registry"
	"github.com/spf13/cobra"
)

var (
	serveAddr string
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the API server",
	Long:  "Start the inference-runtime HTTP API server for remote inference requests.",
	RunE:  runServe,
}

func init() {
	serveCmd.Flags().StringVar(&serveAddr, "addr", ":11434", "Address to listen on")
}

func runServe(_ *cobra.Command, _ []string) error {
	mgr, err := registry.NewModelManager(registry.DefaultBaseDir())
	if err != nil {
		return fmt.Errorf("init model manager: %w", err)
	}

	eng, err := engine.New()
	if err != nil {
		return fmt.Errorf("init engine: %w", err)
	}
	defer eng.Close()

	srv := api.NewServer(eng, mgr, serveAddr)
	return srv.Start()
}
