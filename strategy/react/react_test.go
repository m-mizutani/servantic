package react_test

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/mock"
	"github.com/gollem-dev/gollem/strategy/react"
	"github.com/m-mizutani/gt"
)

func TestNew(t *testing.T) {
	t.Run("creates strategy with defaults", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)
		gt.NotNil(t, strategy)
	})

	t.Run("applies options correctly", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient,
			react.WithMaxIterations(10),
			react.WithMaxRepeatedActions(5),
		)
		gt.NotNil(t, strategy)
	})
}

func TestInit(t *testing.T) {
	ctx := context.Background()

	t.Run("initializes successfully", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		err := strategy.Init(ctx, []gollem.Input{gollem.Text("test")})
		gt.NoError(t, err)
	})
}

func TestTools(t *testing.T) {
	ctx := context.Background()

	t.Run("returns empty tool list", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		tools, err := strategy.Tools(ctx)
		gt.NoError(t, err)
		gt.Equal(t, 0, len(tools))
	})
}

func TestThoughtPhase(t *testing.T) {
	ctx := context.Background()

	t.Run("initial iteration adds thought prompt", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		state := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("solve this problem")},
			Iteration: 0,
		}

		result, resp, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Nil(t, resp)

		// Should have thought prompt + initial input
		if len(result) <= 1 {
			t.Errorf("Expected more than 1 input, got %d", len(result))
		}
	})
}

func TestActionPhase(t *testing.T) {
	ctx := context.Background()

	t.Run("final response without tool calls", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		// Simulate iteration 0
		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, err := strategy.Handle(ctx, state0)
		gt.NoError(t, err)

		// Iteration 1 with final response
		state1 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 1,
			LastResponse: &gollem.Response{
				Texts: []string{"This is the answer"},
			},
		}

		result, resp, err := strategy.Handle(ctx, state1)
		gt.NoError(t, err)
		gt.Nil(t, result)
		gt.NotNil(t, resp)
		gt.Equal(t, "This is the answer", resp.Texts[0])
	})

	t.Run("tool calls trigger observation phase", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		// Simulate iteration 0
		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, err := strategy.Handle(ctx, state0)
		gt.NoError(t, err)

		// Iteration 1 with tool calls
		state1 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			NextInput: []gollem.Input{},
			Iteration: 1,
			LastResponse: &gollem.Response{
				Texts: []string{"Let me search for that"},
				FunctionCalls: []*gollem.FunctionCall{
					{ID: "call-1", Name: "search", Arguments: map[string]any{"query": "test"}},
				},
			},
		}

		result, resp, err := strategy.Handle(ctx, state1)
		gt.NoError(t, err)
		gt.Nil(t, resp)
		gt.Equal(t, 0, len(result))
	})
}

func TestObservationPhase(t *testing.T) {
	ctx := context.Background()

	t.Run("processes tool results successfully", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		// Simulate iterations
		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, _ = strategy.Handle(ctx, state0)

		state1 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			NextInput: []gollem.Input{},
			Iteration: 1,
			LastResponse: &gollem.Response{
				FunctionCalls: []*gollem.FunctionCall{
					{ID: "call-1", Name: "search"},
				},
			},
		}
		_, _, _ = strategy.Handle(ctx, state1)

		// Observation phase with tool results
		state2 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			NextInput: []gollem.Input{
				gollem.FunctionResponse{
					ID:   "call-1",
					Name: "search",
					Data: map[string]any{"result": "found"},
				},
			},
			Iteration: 2,
		}

		result, resp, err := strategy.Handle(ctx, state2)
		gt.NoError(t, err)
		gt.Nil(t, resp)
		if len(result) == 0 {
			t.Error("Expected result to have items")
		}
	})

	// The encoded tool result becomes the observation the model reasons about. An encoding
	// failure used to be ignored, leaving the observation empty instead of reporting it.
	t.Run("reports a tool result that cannot be encoded", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, _ = strategy.Handle(ctx, state0)

		state1 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			NextInput: []gollem.Input{},
			Iteration: 1,
			LastResponse: &gollem.Response{
				FunctionCalls: []*gollem.FunctionCall{
					{ID: "call-1", Name: "search"},
				},
			},
		}
		_, _, _ = strategy.Handle(ctx, state1)

		state2 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			NextInput: []gollem.Input{
				gollem.FunctionResponse{
					ID:   "call-1",
					Name: "search",
					// A channel cannot be encoded as JSON.
					Data: map[string]any{"result": make(chan int)},
				},
			},
			Iteration: 2,
		}

		_, _, err := strategy.Handle(ctx, state2)
		gt.Error(t, err)
	})
}

func TestToolExecutionError(t *testing.T) {
	ctx := context.Background()

	t.Run("handles tool error", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		// Setup
		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, _ = strategy.Handle(ctx, state0)

		state1 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 1,
			LastResponse: &gollem.Response{
				FunctionCalls: []*gollem.FunctionCall{{ID: "call-1", Name: "search"}},
			},
		}
		_, _, _ = strategy.Handle(ctx, state1)

		// Error response
		state2 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			NextInput: []gollem.Input{
				gollem.FunctionResponse{
					ID:    "call-1",
					Name:  "search",
					Error: fmt.Errorf("search failed"),
				},
			},
			Iteration: 2,
		}

		result, resp, err := strategy.Handle(ctx, state2)
		gt.NoError(t, err)
		gt.Nil(t, resp)
		if len(result) == 0 {
			t.Error("Expected result to have items")
		}
	})
}

func TestLoopDetection(t *testing.T) {
	ctx := context.Background()

	t.Run("detects repeated actions", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient,
			react.WithMaxRepeatedActions(2),
		)

		gt.NoError(t, strategy.Init(ctx, nil))

		// Initialize
		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, _ = strategy.Handle(ctx, state0)

		// Repeat the same action
		for i := 1; i <= 3; i++ {
			state := &gollem.StrategyState{
				InitInput: []gollem.Input{gollem.Text("test")},
				Iteration: i,
				LastResponse: &gollem.Response{
					FunctionCalls: []*gollem.FunctionCall{
						{ID: fmt.Sprintf("call-%d", i), Name: "same_tool"},
					},
				},
			}

			_, resp, err := strategy.Handle(ctx, state)
			gt.NoError(t, err)

			if i >= 2 {
				// Should detect loop
				gt.NotNil(t, resp)
				break
			}
		}
	})
}

func TestMaxIterations(t *testing.T) {
	ctx := context.Background()

	t.Run("stops at max iterations", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient,
			react.WithMaxIterations(3),
		)

		gt.NoError(t, strategy.Init(ctx, nil))

		state := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 3,
		}

		result, resp, err := strategy.Handle(ctx, state)
		gt.NoError(t, err)
		gt.Nil(t, result)
		gt.NotNil(t, resp)
	})
}

func TestTraceExport(t *testing.T) {
	ctx := context.Background()

	t.Run("exports trace data", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		// Run through some iterations
		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, _ = strategy.Handle(ctx, state0)

		state1 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 1,
			LastResponse: &gollem.Response{
				Texts: []string{"Done"},
			},
		}
		_, _, _ = strategy.Handle(ctx, state1)

		// Export trace
		trace := strategy.ExportTrace()
		gt.NotNil(t, trace)
		if len(trace.Entries) == 0 {
			t.Error("Expected trace entries")
		}
		gt.Equal(t, "react", trace.Metadata.Strategy)
	})

	t.Run("exports trace as JSON", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("test")},
			Iteration: 0,
		}
		_, _, _ = strategy.Handle(ctx, state0)

		jsonData, err := strategy.ExportTraceJSON()
		gt.NoError(t, err)
		if len(jsonData) == 0 {
			t.Error("Expected JSON data")
		}
	})
}

func TestCompleteScenario(t *testing.T) {
	ctx := context.Background()

	t.Run("complete ReAct cycle", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}
		strategy := react.New(mockClient)

		gt.NoError(t, strategy.Init(ctx, nil))

		// Iteration 0: Initialize
		state0 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("What is 2+2?")},
			Iteration: 0,
		}
		inputs0, resp0, err := strategy.Handle(ctx, state0)
		gt.NoError(t, err)
		gt.Nil(t, resp0)
		if len(inputs0) == 0 {
			t.Error("Expected inputs0 to have items")
		}

		// Iteration 1: Tool call
		state1 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("What is 2+2?")},
			NextInput: []gollem.Input{},
			Iteration: 1,
			LastResponse: &gollem.Response{
				Texts: []string{"Let me calculate that"},
				FunctionCalls: []*gollem.FunctionCall{
					{ID: "calc-1", Name: "calculator", Arguments: map[string]any{"expr": "2+2"}},
				},
			},
		}
		inputs1, resp1, err := strategy.Handle(ctx, state1)
		gt.NoError(t, err)
		gt.Nil(t, resp1)
		gt.Equal(t, 0, len(inputs1))

		// Iteration 2: Observation
		state2 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("What is 2+2?")},
			NextInput: []gollem.Input{
				gollem.FunctionResponse{
					ID:   "calc-1",
					Name: "calculator",
					Data: map[string]any{"result": 4},
				},
			},
			Iteration: 2,
		}
		inputs2, resp2, err := strategy.Handle(ctx, state2)
		gt.NoError(t, err)
		gt.Nil(t, resp2)
		if len(inputs2) == 0 {
			t.Error("Expected inputs2 to have items")
		}

		// Iteration 3: Final answer
		state3 := &gollem.StrategyState{
			InitInput: []gollem.Input{gollem.Text("What is 2+2?")},
			NextInput: []gollem.Input{},
			Iteration: 3,
			LastResponse: &gollem.Response{
				Texts: []string{"The answer is 4"},
			},
		}
		inputs3, resp3, err := strategy.Handle(ctx, state3)
		gt.NoError(t, err)
		gt.Nil(t, inputs3)
		gt.NotNil(t, resp3)
		gt.Equal(t, "The answer is 4", resp3.Texts[0])

		// Verify trace
		trace := strategy.ExportTrace()
		if len(trace.Entries) == 0 {
			t.Error("Expected trace entries")
		}
	})
}

// CalculatorTool is a simple calculator tool for testing
type CalculatorTool struct{}

func (t *CalculatorTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "calculator",
		Description: "Performs basic arithmetic operations (add, subtract, multiply, divide)",
		Parameters: map[string]*gollem.Parameter{
			"operation": {
				Type:        gollem.TypeString,
				Description: "The operation to perform: add, subtract, multiply, or divide",
				Enum:        []string{"add", "subtract", "multiply", "divide"},
				Required:    true,
			},
			"a": {
				Type:        gollem.TypeNumber,
				Description: "First number",
				Required:    true,
			},
			"b": {
				Type:        gollem.TypeNumber,
				Description: "Second number",
				Required:    true,
			},
		},
	}
}

func (t *CalculatorTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	operation := args["operation"].(string)
	a := args["a"].(float64)
	b := args["b"].(float64)

	var result float64
	switch operation {
	case "add":
		result = a + b
	case "subtract":
		result = a - b
	case "multiply":
		result = a * b
	case "divide":
		if b == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		result = a / b
	default:
		return nil, fmt.Errorf("unknown operation: %s", operation)
	}

	return map[string]any{
		"result": result,
	}, nil
}

// RandomNumberTool generates a random number
type RandomNumberTool struct{}

func (t *RandomNumberTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "random_number",
		Description: "Generates a random number within a specified range",
		Parameters: map[string]*gollem.Parameter{
			"min": {
				Type:        gollem.TypeNumber,
				Description: "Minimum value of the range",
				Required:    true,
			},
			"max": {
				Type:        gollem.TypeNumber,
				Description: "Maximum value of the range",
				Required:    true,
			},
		},
	}
}

func (t *RandomNumberTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	min := int(args["min"].(float64))
	max := int(args["max"].(float64))

	if min >= max {
		return nil, fmt.Errorf("min must be less than max")
	}

	randomNum := rand.Intn(max-min) + min
	return map[string]any{
		"number": randomNum,
	}, nil
}

// FileSystemTool simulates a file system for exploration
// Structure: /users/{user}/profile.json
type FileSystemTool struct {
	data map[string]string
}

func NewFileSystemTool() *FileSystemTool {
	// Create a puzzle requiring multi-step reasoning where each step depends on the previous:
	// Step 1: Find which directory contains the key
	// Step 2: Use the key to determine which file to read
	// Step 3: Get the final answer from that file
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Random values that create dependencies between steps
	keyNumber := r.Intn(3) + 1                                                                    // 1, 2, or 3
	targetFile := fmt.Sprintf("secret%d.txt", keyNumber)                                          // secret1.txt, secret2.txt, or secret3.txt
	finalAnswer := fmt.Sprintf("%d%d%d", r.Intn(9000)+1000, r.Intn(9000)+1000, r.Intn(9000)+1000) // Random 12-digit number

	return &FileSystemTool{
		data: map[string]string{
			"/":                                  "step1,step2,step3",
			"/step1":                             "instructions.txt",
			"/step2":                             "key.txt,secret1.txt,secret2.txt,secret3.txt",
			"/step3":                             "readme.md",
			"/step1/instructions.txt":            "To find the answer, first read /step2/key.txt to discover which secret file contains the answer.",
			"/step2/key.txt":                     fmt.Sprintf(`The answer is in the file: %s`, targetFile),
			"/step2/secret1.txt":                 "This is a decoy file.",
			"/step2/secret2.txt":                 "This is also a decoy.",
			"/step2/secret3.txt":                 "Another decoy file.",
			fmt.Sprintf("/step2/%s", targetFile): fmt.Sprintf(`ANSWER: %s`, finalAnswer),
			"/step3/readme.md":                   "This directory is empty, just a distraction.",
		},
	}
}

func (t *FileSystemTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "list_directory",
		Description: "Lists the contents of a directory. Returns comma-separated list of entries.",
		Parameters: map[string]*gollem.Parameter{
			"path": {
				Type:        gollem.TypeString,
				Description: "The directory path to list (e.g., '/', '/users', '/users/alice')",
				Required:    true,
			},
		},
	}
}

func (t *FileSystemTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	path := args["path"].(string)
	content, ok := t.data[path]
	if !ok {
		return nil, fmt.Errorf("path not found: %s", path)
	}
	return map[string]any{
		"contents": content,
	}, nil
}

// ReadFileTool reads file contents
type ReadFileTool struct {
	fs *FileSystemTool
}

func (t *ReadFileTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "read_file",
		Description: "Reads the contents of a file.",
		Parameters: map[string]*gollem.Parameter{
			"path": {
				Type:        gollem.TypeString,
				Description: "The file path to read (e.g., '/users/alice/profile.json')",
				Required:    true,
			},
		},
	}
}

func (t *ReadFileTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	path := args["path"].(string)
	content, ok := t.fs.data[path]
	if !ok {
		return nil, fmt.Errorf("file not found: %s", path)
	}
	// Only return content if it looks like a file (contains a dot)
	if !containsChar(path, '.') {
		return nil, fmt.Errorf("path is a directory, not a file: %s", path)
	}
	return map[string]any{
		"content": content,
	}, nil
}

func containsChar(s string, c rune) bool {
	for _, ch := range s {
		if ch == c {
			return true
		}
	}
	return false
}

// TestReActWithRealLLM tests ReAct strategy with real LLM clients
func TestReActWithRealLLM(t *testing.T) {
	t.Parallel()

	testFn := func(t *testing.T, newClient func(t *testing.T) (gollem.LLMClient, error)) {
		client, err := newClient(t)
		gt.NoError(t, err)

		// Create file system tools for exploration
		fsTool := NewFileSystemTool()
		readTool := &ReadFileTool{fs: fsTool}

		// Extract the expected answer
		var expectedAnswer string
		for _, content := range fsTool.data {
			var ans string
			if n, _ := fmt.Sscanf(content, "ANSWER: %s", &ans); n == 1 {
				expectedAnswer = ans
				break
			}
		}
		t.Logf("Expected answer: %s", expectedAnswer)

		strategy := react.New(client,
			react.WithMaxIterations(20),
			react.WithMaxRepeatedActions(10),
		)

		agent := gollem.New(client,
			gollem.WithStrategy(strategy),
			gollem.WithTools(fsTool, readTool),
			gollem.WithLoopLimit(20),
		)

		// Test: Multi-step reasoning task requiring chained tool usage
		ctx := context.Background()
		resp, err := agent.Execute(ctx, gollem.Text(`Find the secret code hidden in the filesystem. You must use the list_directory tool to explore directories and the read_file tool to read file contents. Start from the root directory "/" and systematically explore until you find the secret code.`))
		// Stop here on failure: the assertions below dereference resp, so continuing
		// would panic and abort every other test in this package.
		gt.NoError(t, err).Required()
		gt.V(t, resp).NotNil().Required()

		// Export and verify trace
		trace := strategy.ExportTrace()
		gt.V(t, trace).NotNil().Required()

		// Verify response exists
		gt.N(t, len(resp.Texts)).Greater(0)
		responseText := resp.Texts[0]
		gt.S(t, responseText).NotEqual("")

		// === ReAct Validation ===

		// 1. Must have at least one TAO cycle
		gt.N(t, len(trace.Entries)).GreaterOrEqual(1)

		// 2. Strategy must be react
		gt.Equal(t, trace.Metadata.Strategy, "react")

		// 3. Verify TAO structure
		for i, entry := range trace.Entries {
			t.Logf("=== Entry %d ===", i)

			// Must have Thought
			gt.NotNil(t, entry.Thought)
			gt.S(t, entry.Thought.Content).NotEqual("")
			t.Logf("Thought: %s", entry.Thought.Content)

			// Must have Action
			gt.NotNil(t, entry.Action)
			t.Logf("Action Type: %s", entry.Action.Type)

			if entry.Action.Type == react.ActionTypeToolCall {
				for _, call := range entry.Action.ToolCalls {
					t.Logf("Tool: %s", call.Name)
				}

				// Must have Observation
				gt.NotNil(t, entry.Observation)
			}
		}

		// 4. Verify response contains the expected answer
		if !strings.Contains(responseText, expectedAnswer) {
			t.Logf("Expected to find '%s' in response, but got: %s", expectedAnswer, responseText)
		}
		gt.S(t, responseText).Contains(expectedAnswer)

		// Log summary
		t.Logf("=== Summary ===")
		t.Logf("Response: %s", responseText)
		t.Logf("Tool calls made: %d", trace.Summary.ToolCallsCount)
		t.Logf("Total iterations: %d", trace.Summary.TotalIterations)
		t.Logf("TAO entries: %d", len(trace.Entries))
	}

	t.Run("OpenAI", func(t *testing.T) {
		t.Parallel()
		apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
		if !ok {
			t.Skip("TEST_OPENAI_API_KEY is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return openai.New(context.Background(), apiKey)
		})
	})

	t.Run("Claude", func(t *testing.T) {
		apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
		if !ok {
			t.Skip("TEST_CLAUDE_API_KEY is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return claude.New(context.Background(), apiKey)
		})
	})

	t.Run("Gemini", func(t *testing.T) {
		t.Parallel()
		projectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
		if !ok {
			t.Skip("TEST_GCP_PROJECT_ID is not set")
		}
		location, ok := os.LookupEnv("TEST_GCP_LOCATION")
		if !ok {
			t.Skip("TEST_GCP_LOCATION is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return gemini.New(context.Background(), projectID, location)
		})
	})
}
