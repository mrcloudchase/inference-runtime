package cmd

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/cloudchase/inference-runtime/engine"
	"github.com/cloudchase/inference-runtime/registry"
	"github.com/spf13/cobra"
)

var runCmd = &cobra.Command{
	Use:   "run <model> [prompt]",
	Short: "Run inference on a model",
	Long: `Run a model for text generation. If a prompt is provided as an argument,
generate a response and exit. Otherwise, start an interactive REPL.

The model argument can be a registered model name or a path to a GGUF file.`,
	Args: cobra.MinimumNArgs(1),
	RunE: runRun,
}

func runRun(cmd *cobra.Command, args []string) error {
	modelArg := args[0]
	prompt := ""
	if len(args) > 1 {
		prompt = strings.Join(args[1:], " ")
	}

	mgr, err := registry.NewModelManager(registry.DefaultBaseDir())
	if err != nil {
		return fmt.Errorf("init model manager: %w", err)
	}

	modelPath, err := mgr.ResolveModelPath(modelArg)
	if err != nil {
		return err
	}

	eng, err := engine.New()
	if err != nil {
		return err
	}
	defer eng.Close()

	fmt.Fprintf(os.Stderr, "Loading model: %s\n", modelPath)
	if err := eng.LoadModel(modelPath); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "Model loaded.")

	opts := engine.DefaultOptions()

	// Single-shot mode: generate and exit.
	if prompt != "" {
		return generateAndPrint(eng, prompt, opts)
	}

	// Interactive REPL mode.
	return repl(eng, opts)
}

func generateAndPrint(eng *engine.Engine, prompt string, opts engine.GenerateOptions) error {
	return eng.GenerateStream(prompt, opts, func(token string) bool {
		fmt.Print(token)
		return true
	})
}

func repl(eng *engine.Engine, opts engine.GenerateOptions) error {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Print(">>> ")

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			fmt.Print(">>> ")
			continue
		}

		switch strings.ToLower(line) {
		case "/exit", "/quit", "/bye":
			fmt.Println("Goodbye.")
			return nil
		case "/reset":
			if err := eng.Reset(); err != nil {
				fmt.Fprintf(os.Stderr, "Reset error: %v\n", err)
			} else {
				fmt.Fprintln(os.Stderr, "Context reset.")
			}
			fmt.Print(">>> ")
			continue
		case "/help":
			fmt.Println("Commands:")
			fmt.Println("  /exit, /quit, /bye  - Exit the REPL")
			fmt.Println("  /reset              - Reset context")
			fmt.Println("  /help               - Show this help")
			fmt.Println("  <text>              - Generate a response")
			fmt.Print(">>> ")
			continue
		}

		if err := generateAndPrint(eng, line, opts); err != nil {
			fmt.Fprintf(os.Stderr, "\nGeneration error: %v\n", err)
		}
		fmt.Println()
		fmt.Print(">>> ")
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("reading stdin: %w", err)
	}
	return nil
}
