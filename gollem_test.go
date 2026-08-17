package gollem_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
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
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

// RandomNumberTool is a tool that generates a random number within a specified range
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

func TestGollemWithTool(t *testing.T) {
	t.Run("tool execution", func(t *testing.T) {
		callCount := 0
		toolCalled := false

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++

						// First call: return tool call
						if callCount == 1 {
							return &gollem.Response{
								Texts: []string{"I'll generate a random number for you."},
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:   "call_random_1",
										Name: "random_number",
										Arguments: map[string]any{
											"min": float64(1),
											"max": float64(100),
										},
									},
								},
							}, nil
						}

						// Second call: handle tool response
						if callCount == 2 {
							// Verify we received a function response
							if len(input) > 0 {
								if funcResp, ok := input[0].(gollem.FunctionResponse); ok {
									gt.Equal(t, "call_random_1", funcResp.ID)
									gt.Equal(t, "random_number", funcResp.Name)
									// Verify the response contains a number
									if result, ok := funcResp.Data["number"]; ok {
										var num int
										switch v := result.(type) {
										case int:
											num = v
										case float64:
											num = int(v)
										}
										if num >= 1 && num <= 100 {
											toolCalled = true
										}
									}
								}
							}
							// End the conversation
							return &gollem.Response{
								Texts: []string{"Tool execution completed"},
							}, nil
						}

						return &gollem.Response{
							Texts: []string{"unexpected call"},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		tool := &RandomNumberTool{}
		s := gollem.New(mockClient,
			gollem.WithTools(tool),
			gollem.WithLoopLimit(5),
		)

		result, err := s.Execute(t.Context(), gollem.Text("Generate a random number between 1 and 100."))
		gt.NoError(t, err)
		// Check that we got some result (either from strategy or default behavior)
		_ = result
		gt.True(t, toolCalled)
		gt.Equal(t, 2, callCount)
	})

	t.Run("tool middleware", func(t *testing.T) {
		middlewareCalled := false
		var capturedToolCalls []*gollem.FunctionCall

		// Tool middleware that captures tool calls
		toolMiddleware := func(next gollem.ToolHandler) gollem.ToolHandler {
			return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
				middlewareCalled = true
				capturedToolCalls = append(capturedToolCalls, req.Tool)

				// Call the next handler
				resp, err := next(ctx, req)
				if err != nil {
					return nil, err
				}

				// Modify the response
				if resp.Result != nil {
					if result, ok := resp.Result["number"]; ok {
						resp.Result["middleware_processed"] = true
						resp.Result["original_number"] = result
					}
				}

				return resp, nil
			}
		}

		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						// First call: return tool call
						if callCount == 1 {
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:   "test_call",
										Name: "random_number",
										Arguments: map[string]any{
											"min": float64(1),
											"max": float64(10),
										},
									},
								},
							}, nil
						}
						// Second call: end the conversation
						return &gollem.Response{
							Texts: []string{"Done"},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		tool := &RandomNumberTool{}
		s := gollem.New(mockClient,
			gollem.WithTools(tool),
			gollem.WithToolMiddleware(toolMiddleware),
			gollem.WithLoopLimit(5),
		)

		_, err := s.Execute(t.Context(), gollem.Text("test"))
		gt.NoError(t, err)
		gt.True(t, middlewareCalled)
		gt.Equal(t, 1, len(capturedToolCalls))
		gt.Equal(t, "random_number", capturedToolCalls[0].Name)
	})
}

func TestPromptCachePropagation(t *testing.T) {
	runTest := func(enabled bool) func(t *testing.T) {
		return func(t *testing.T) {
			var captured bool
			mockClient := &mock.LLMClientMock{
				NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
					// Reconstruct the session config the agent assembled to observe
					// whether the prompt-cache flag was threaded through.
					cfg := gollem.NewSessionConfig(options...)
					captured = cfg.PromptCache()
					return &mock.SessionMock{
						GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
							return &gollem.Response{Texts: []string{"done"}}, nil
						},
					}, nil
				},
			}

			opts := []gollem.Option{gollem.WithLoopLimit(2)}
			if enabled {
				opts = append(opts, gollem.WithPromptCache(true))
			}
			s := gollem.New(mockClient, opts...)
			_, err := s.Execute(t.Context(), gollem.Text("hi"))
			gt.NoError(t, err)
			gt.Equal(t, enabled, captured)
		}
	}

	t.Run("enabled propagates to session", runTest(true))
	t.Run("disabled by default", runTest(false))
}

// mockTool is a mock implementation of gollem.Tool
type mockTool struct {
	spec gollem.ToolSpec
	run  func(ctx context.Context, args map[string]any) (map[string]any, error)
}

func (t *mockTool) Spec() gollem.ToolSpec {
	return t.spec
}

func (t *mockTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return t.run(ctx, args)
}

// newMockClient creates a new LLMClientMock with the given GenerateFunc
func newMockClient(generateContentFunc func(ctx context.Context, input ...gollem.Input) (*gollem.Response, error)) *mock.LLMClientMock {
	return &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			mockSession := &mock.SessionMock{
				GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
					response, err := generateContentFunc(ctx, input...)
					if err != nil {
						return nil, err
					}
					return response, nil
				},
			}
			return mockSession, nil
		},
	}
}

func TestGollemWithOptions(t *testing.T) {
	t.Run("WithLoopLimit", func(t *testing.T) {
		loopCount := 0
		mockClient := newMockClient(func(ctx context.Context, input ...gollem.Input) (*gollem.Response, error) {
			loopCount++
			return &gollem.Response{
				Texts: []string{"test response"},
				FunctionCalls: []*gollem.FunctionCall{
					{
						Name: "test_tool",
						Arguments: map[string]any{
							"arg1": "value1",
						},
					},
				},
			}, nil
		})

		tool := &mock.ToolMock{
			SpecFunc: func() gollem.ToolSpec {
				return gollem.ToolSpec{
					Name:        "test_tool",
					Description: "A test tool",
				}
			},
			RunFunc: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "test"}, nil
			},
		}

		s := gollem.New(mockClient, gollem.WithLoopLimit(10), gollem.WithTools(tool))
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrLoopLimitExceeded))
		gt.Equal(t, loopCount, 10)
	})

	t.Run("WithSystemPrompt", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				gt.Equal(t, cfg.SystemPrompt(), "system prompt")
				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						// Return response based on input
						if len(input) > 0 {
							if text, ok := input[0].(gollem.Text); ok {
								if strings.Contains(string(text), "test message") {
									// Return response with tool call
									return &gollem.Response{
										FunctionCalls: []*gollem.FunctionCall{
											{
												ID:        "test_call_1",
												Name:      "test_tool",
												Arguments: map[string]any{},
											},
										},
									}, nil
								}
							}
						}

						// Handle function responses
						if len(input) > 0 {
							if _, ok := input[0].(gollem.FunctionResponse); ok {
								// Return response with no tool calls to end the loop
								return &gollem.Response{
									Texts: []string{"Task completed"},
								}, nil
							}
						}

						return &gollem.Response{
							Texts: []string{"test response"},
							FunctionCalls: []*gollem.FunctionCall{
								{
									Name:      "respond_to_user",
									Arguments: map[string]any{},
								},
							},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		s := gollem.New(mockClient, gollem.WithSystemPrompt("system prompt"))
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
	})

	t.Run("WithTools", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				tools := cfg.Tools()
				// Should have test_tool only
				gt.Equal(t, len(tools), 1)
				toolNames := make(map[string]bool)
				for _, tool := range tools {
					toolNames[tool.Spec().Name] = true
				}
				gt.True(t, toolNames["test_tool"])

				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						// Return response based on input
						if len(input) > 0 {
							if text, ok := input[0].(gollem.Text); ok {
								if strings.Contains(string(text), "test message") {
									// Return response with tool call
									return &gollem.Response{
										FunctionCalls: []*gollem.FunctionCall{
											{
												ID:        "test_call_1",
												Name:      "test_tool",
												Arguments: map[string]any{},
											},
										},
									}, nil
								}
							}
						}

						// Handle function responses
						if len(input) > 0 {
							if _, ok := input[0].(gollem.FunctionResponse); ok {
								// Return response with no tool calls to end the loop
								return &gollem.Response{
									Texts: []string{"Task completed"},
								}, nil
							}
						}

						return &gollem.Response{
							Texts: []string{"test response"},
							FunctionCalls: []*gollem.FunctionCall{
								{
									Name:      "respond_to_user",
									Arguments: map[string]any{},
								},
							},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		tool := &mockTool{
			spec: gollem.ToolSpec{
				Name:        "test_tool",
				Description: "A test tool",
			},
			run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "test"}, nil
			},
		}
		s := gollem.New(mockClient, gollem.WithTools(tool), gollem.WithLoopLimit(5))
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
	})

	t.Run("WithToolSets", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				tools := cfg.Tools()
				// Should have test_tool from ToolSet only
				gt.Equal(t, len(tools), 1)

				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						// Return response based on input
						if len(input) > 0 {
							if text, ok := input[0].(gollem.Text); ok {
								if strings.Contains(string(text), "test message") {
									// Return response with tool call
									return &gollem.Response{
										FunctionCalls: []*gollem.FunctionCall{
											{
												ID:        "test_call_1",
												Name:      "test_tool",
												Arguments: map[string]any{},
											},
										},
									}, nil
								}
							}
						}

						// Handle function responses
						if len(input) > 0 {
							if _, ok := input[0].(gollem.FunctionResponse); ok {
								// Return response with no tool calls to end the loop
								return &gollem.Response{
									Texts: []string{"Task completed"},
								}, nil
							}
						}

						return &gollem.Response{
							Texts: []string{"test response"},
							FunctionCalls: []*gollem.FunctionCall{
								{
									Name:      "respond_to_user",
									Arguments: map[string]any{},
								},
							},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		toolSet := &mockToolSet{
			specs: []gollem.ToolSpec{
				{
					Name:        "test_tool",
					Description: "A test tool",
				},
			},
			run: func(ctx context.Context, name string, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "test"}, nil
			},
		}
		s := gollem.New(mockClient, gollem.WithToolSets(toolSet), gollem.WithLoopLimit(5))
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
	})

	t.Run("WithResponseMode", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				// Check session options to determine if this is for streaming
				cfg := gollem.NewSessionConfig(options...)
				isStreamingSession := cfg.SystemPrompt() == "" // Main session has no specific system prompt for streaming

				mockSession := &mock.SessionMock{
					StreamFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
						ch := make(chan *gollem.Response)
						go func() {
							defer close(ch)
							// Only handle streaming for the main session
							if isStreamingSession {
								ch <- &gollem.Response{
									Texts: []string{"test response 1"},
								}
								ch <- &gollem.Response{
									Texts: []string{"test response 2"},
								}
								ch <- &gollem.Response{
									Texts: []string{"test response 3"},
								}
							}
						}()
						return ch, nil
					},
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						// Return response based on input
						if len(input) > 0 {
							if text, ok := input[0].(gollem.Text); ok {
								if strings.Contains(string(text), "test message") {
									// Return response with tool call
									return &gollem.Response{
										FunctionCalls: []*gollem.FunctionCall{
											{
												ID:        "test_call_1",
												Name:      "test_tool",
												Arguments: map[string]any{},
											},
										},
									}, nil
								}
							}
						}
						// Handle function responses
						if len(input) > 0 {
							if _, ok := input[0].(gollem.FunctionResponse); ok {
								// Return response with no tool calls to end the loop
								return &gollem.Response{
									Texts: []string{"Task completed"},
								}, nil
							}
						}
						return &gollem.Response{}, nil
					},
				}
				return mockSession, nil
			},
		}

		s := gollem.New(mockClient,
			gollem.WithResponseMode(gollem.ResponseModeStreaming),
		)
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
		// Test completes successfully with streaming mode
	})

	t.Run("WithLogger", func(t *testing.T) {
		var logOutput strings.Builder
		logger := slog.New(slog.NewTextHandler(&logOutput, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		}))

		mockClient := newMockClient(func(ctx context.Context, input ...gollem.Input) (*gollem.Response, error) {
			return &gollem.Response{
				Texts: []string{"test response"},
			}, nil
		})

		s := gollem.New(mockClient, gollem.WithLogger(logger), gollem.WithLoopLimit(5))
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)

		logContent := logOutput.String()
		gt.True(t, len(logContent) > 0)
	})

	t.Run("CombineOptions", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				gt.Equal(t, cfg.SystemPrompt(), "system prompt")
				// Should have test_tool only
				gt.Equal(t, len(cfg.Tools()), 1)
				toolNames := make(map[string]bool)
				for _, tool := range cfg.Tools() {
					toolNames[tool.Spec().Name] = true
				}
				gt.True(t, toolNames["test_tool"])

				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						// Return response based on input
						if len(input) > 0 {
							if text, ok := input[0].(gollem.Text); ok {
								if strings.Contains(string(text), "test message") {
									// Return response with tool call
									return &gollem.Response{
										FunctionCalls: []*gollem.FunctionCall{
											{
												ID:        "test_call_1",
												Name:      "test_tool",
												Arguments: map[string]any{},
											},
										},
									}, nil
								}
							}
						}

						// Handle function responses
						if len(input) > 0 {
							if _, ok := input[0].(gollem.FunctionResponse); ok {
								// Return response with no tool calls to end the loop
								return &gollem.Response{
									Texts: []string{"Task completed"},
								}, nil
							}
						}

						return &gollem.Response{
							Texts: []string{"test response"},
							FunctionCalls: []*gollem.FunctionCall{
								{
									Name:      "respond_to_user",
									Arguments: map[string]any{},
								},
							},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		tool := &mockTool{
			spec: gollem.ToolSpec{
				Name:        "test_tool",
				Description: "A test tool",
			},
			run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "test"}, nil
			},
		}
		logger := slog.New(slog.NewTextHandler(io.Discard, nil))

		s := gollem.New(mockClient,
			gollem.WithLoopLimit(5),
			gollem.WithSystemPrompt("system prompt"),
			gollem.WithTools(tool),
			gollem.WithResponseMode(gollem.ResponseModeBlocking),
			gollem.WithLogger(logger),
		)
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
	})

	t.Run("WithToolMiddleware", func(t *testing.T) {
		t.Parallel()

		// Create tool middleware that tracks execution
		middlewareExecuted := false
		testMiddleware := func(next gollem.ToolHandler) gollem.ToolHandler {
			return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
				middlewareExecuted = true
				return next(ctx, req)
			}
		}

		// Create agent with tool middleware
		agent := gollem.New(nil,
			gollem.WithToolMiddleware(testMiddleware),
		)

		gt.NotNil(t, agent)
		// Middleware configuration verification - agent accepts tool middleware
		_ = middlewareExecuted // Reserved for future execution testing
	})

	t.Run("WithContentType", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				// Verify that ContentType was passed to the session
				gt.Equal(t, gollem.ContentTypeJSON, cfg.ContentType())

				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						return &gollem.Response{
							Texts: []string{`{"result": "success"}`},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		s := gollem.New(mockClient,
			gollem.WithContentType(gollem.ContentTypeJSON),
			gollem.WithLoopLimit(5),
		)
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
	})

	t.Run("WithResponseSchema", func(t *testing.T) {
		schema := &gollem.Parameter{
			Type:        gollem.TypeObject,
			Description: "Test response schema",
			Properties: map[string]*gollem.Parameter{
				"result": {
					Type:        gollem.TypeString,
					Description: "Result field",
					Required:    true,
				},
			},
		}

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				// Verify that ResponseSchema was passed to the session
				gt.NotNil(t, cfg.ResponseSchema())
				gt.Equal(t, "Test response schema", cfg.ResponseSchema().Description)
				gt.Equal(t, gollem.TypeObject, cfg.ResponseSchema().Type)

				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						return &gollem.Response{
							Texts: []string{`{"result": "success"}`},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		s := gollem.New(mockClient,
			gollem.WithResponseSchema(schema),
			gollem.WithLoopLimit(5),
		)
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
	})

	t.Run("WithContentTypeAndResponseSchema", func(t *testing.T) {
		schema := &gollem.Parameter{
			Type:        gollem.TypeObject,
			Description: "Combined test schema",
			Properties: map[string]*gollem.Parameter{
				"status": {
					Type:        gollem.TypeString,
					Description: "Status field",
					Required:    true,
				},
				"message": {
					Type:        gollem.TypeString,
					Description: "Message field",
				},
			},
		}

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				cfg := gollem.NewSessionConfig(options...)
				// Verify both ContentType and ResponseSchema were passed
				gt.Equal(t, gollem.ContentTypeJSON, cfg.ContentType())
				gt.NotNil(t, cfg.ResponseSchema())
				gt.Equal(t, "Combined test schema", cfg.ResponseSchema().Description)

				mockSession := &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						return &gollem.Response{
							Texts: []string{`{"status": "ok", "message": "test"}`},
						}, nil
					},
				}
				return mockSession, nil
			},
		}

		s := gollem.New(mockClient,
			gollem.WithContentType(gollem.ContentTypeJSON),
			gollem.WithResponseSchema(schema),
			gollem.WithLoopLimit(5),
		)
		_, err := s.Execute(t.Context(), gollem.Text("test message"))
		gt.NoError(t, err)
	})
}

// mockToolSet is a mock implementation of gollem.ToolSet
type mockToolSet struct {
	specs []gollem.ToolSpec
	run   func(ctx context.Context, name string, args map[string]any) (map[string]any, error)
}

func (t *mockToolSet) Specs(ctx context.Context) ([]gollem.ToolSpec, error) {
	return t.specs, nil
}

func (t *mockToolSet) Run(ctx context.Context, name string, args map[string]any) (map[string]any, error) {
	return t.run(ctx, name, args)
}

func TestExecuteWithExecuteResponse(t *testing.T) {
	t.Run("strategy returns ExecuteResponse", func(t *testing.T) {
		// Create a strategy that immediately returns an ExecuteResponse
		strategy := &mock.StrategyMock{
			InitFunc: func(ctx context.Context, inputs []gollem.Input) error {
				return nil
			},
			HandleFunc: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				return nil, gollem.NewExecuteResponse("Test conclusion"), nil
			},
			ToolsFunc: func(ctx context.Context) ([]gollem.Tool, error) {
				return []gollem.Tool{}, nil
			},
		}

		llmClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{}, nil
			},
		}
		agent := gollem.New(llmClient, gollem.WithStrategy(strategy))
		result, err := agent.Execute(context.Background(), gollem.Text("test"))

		gt.NoError(t, err)
		gt.NotNil(t, result)
		gt.Equal(t, "Test conclusion", result.String())
	})

	t.Run("strategy returns both ExecuteResponse and Input with warning", func(t *testing.T) {
		var logOutput strings.Builder
		logger := slog.New(slog.NewTextHandler(&logOutput, &slog.HandlerOptions{Level: slog.LevelWarn}))

		// Strategy that returns both ExecuteResponse and Input
		strategy := &mock.StrategyMock{
			InitFunc: func(ctx context.Context, inputs []gollem.Input) error {
				return nil
			},
			HandleFunc: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				return []gollem.Input{gollem.Text("ignored")},
					gollem.NewExecuteResponse("conclusion"),
					nil
			},
			ToolsFunc: func(ctx context.Context) ([]gollem.Tool, error) {
				return []gollem.Tool{}, nil
			},
		}

		llmClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{}, nil
			},
		}
		agent := gollem.New(llmClient,
			gollem.WithStrategy(strategy),
			gollem.WithLogger(logger))

		result, err := agent.Execute(context.Background(), gollem.Text("test"))

		gt.NoError(t, err)
		gt.NotNil(t, result)
		gt.Equal(t, "conclusion", result.String())
		// Check that warning was logged
		gt.True(t, strings.Contains(logOutput.String(), "Strategy returned both ExecuteResponse and Input"))
	})

	t.Run("strategy returns nil for both", func(t *testing.T) {
		strategy := &mock.StrategyMock{
			InitFunc: func(ctx context.Context, inputs []gollem.Input) error {
				return nil
			},
			HandleFunc: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				return nil, nil, nil
			},
			ToolsFunc: func(ctx context.Context) ([]gollem.Tool, error) {
				return []gollem.Tool{}, nil
			},
		}

		llmClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{}, nil
			},
		}
		agent := gollem.New(llmClient, gollem.WithStrategy(strategy))
		result, err := agent.Execute(context.Background(), gollem.Text("test"))

		gt.NoError(t, err)
		gt.Nil(t, result)
	})

	t.Run("strategy tool name conflict detection", func(t *testing.T) {
		conflictingToolName := "conflicting_tool"

		// Create a regular tool with a specific name
		userTool := &mockTool{
			spec: gollem.ToolSpec{
				Name:        conflictingToolName,
				Description: "User provided tool",
			},
			run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"source": "user"}, nil
			},
		}

		// Create a strategy that provides a tool with the same name
		strategy := &mock.StrategyMock{
			InitFunc: func(ctx context.Context, inputs []gollem.Input) error {
				return nil
			},
			HandleFunc: func(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
				return nil, gollem.NewExecuteResponse("Should not reach here"), nil
			},
			ToolsFunc: func(ctx context.Context) ([]gollem.Tool, error) {
				strategyTool := &mockTool{
					spec: gollem.ToolSpec{
						Name:        conflictingToolName,
						Description: "Strategy provided tool",
					},
					run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
						return map[string]any{"source": "strategy"}, nil
					},
				}
				return []gollem.Tool{strategyTool}, nil
			},
		}

		llmClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{}, nil
			},
		}
		agent := gollem.New(llmClient,
			gollem.WithTools(userTool),
			gollem.WithStrategy(strategy))

		// Execute should fail with tool name conflict error
		_, err := agent.Execute(context.Background(), gollem.Text("test"))

		gt.Error(t, err)
		gt.True(t, errors.Is(err, gollem.ErrToolNameConflict))
	})
}

func TestArgsValidation(t *testing.T) {
	t.Run("invalid args returns validation error to LLM", func(t *testing.T) {
		callCount := 0
		var receivedError error

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							// LLM sends tool call with wrong type args
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:   "call_1",
										Name: "search",
										Arguments: map[string]any{
											"query": 123, // wrong type: should be string
										},
									},
								},
							}, nil
						}
						// Second call: LLM receives validation error and should see it
						if callCount == 2 {
							if len(input) > 0 {
								if funcResp, ok := input[0].(gollem.FunctionResponse); ok {
									receivedError = funcResp.Error
								}
							}
							return &gollem.Response{
								Texts: []string{"Done"},
							}, nil
						}
						return &gollem.Response{Texts: []string{"unexpected"}}, nil
					},
				}, nil
			},
		}

		toolRunCalled := false
		tool := &mockTool{
			spec: gollem.ToolSpec{
				Name: "search",
				Parameters: map[string]*gollem.Parameter{
					"query": {Type: gollem.TypeString, Required: true},
				},
			},
			run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				toolRunCalled = true
				return map[string]any{"result": "ok"}, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithTools(tool), gollem.WithLoopLimit(5))
		_, err := agent.Execute(t.Context(), gollem.Text("search something"))
		gt.NoError(t, err)
		gt.Equal(t, 2, callCount)
		gt.False(t, toolRunCalled) // tool.Run should NOT have been called
		gt.NotNil(t, receivedError)
		gt.S(t, receivedError.Error()).Contains("search")
	})

	t.Run("valid args allows tool execution", func(t *testing.T) {
		callCount := 0

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:   "call_1",
										Name: "search",
										Arguments: map[string]any{
											"query": "hello",
										},
									},
								},
							}, nil
						}
						return &gollem.Response{Texts: []string{"Done"}}, nil
					},
				}, nil
			},
		}

		toolRunCalled := false
		tool := &mockTool{
			spec: gollem.ToolSpec{
				Name: "search",
				Parameters: map[string]*gollem.Parameter{
					"query": {Type: gollem.TypeString, Required: true},
				},
			},
			run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				toolRunCalled = true
				return map[string]any{"result": "found"}, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithTools(tool), gollem.WithLoopLimit(5))
		_, err := agent.Execute(t.Context(), gollem.Text("search something"))
		gt.NoError(t, err)
		gt.True(t, toolRunCalled)
	})

	t.Run("missing required arg returns validation error to LLM", func(t *testing.T) {
		callCount := 0
		var receivedError error

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							// LLM sends tool call without required param
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:        "call_1",
										Name:      "search",
										Arguments: map[string]any{}, // missing "query"
									},
								},
							}, nil
						}
						if callCount == 2 {
							if len(input) > 0 {
								if funcResp, ok := input[0].(gollem.FunctionResponse); ok {
									receivedError = funcResp.Error
								}
							}
							return &gollem.Response{Texts: []string{"Done"}}, nil
						}
						return &gollem.Response{Texts: []string{"unexpected"}}, nil
					},
				}, nil
			},
		}

		toolRunCalled := false
		tool := &mockTool{
			spec: gollem.ToolSpec{
				Name: "search",
				Parameters: map[string]*gollem.Parameter{
					"query": {Type: gollem.TypeString, Required: true},
				},
			},
			run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				toolRunCalled = true
				return map[string]any{"result": "ok"}, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithTools(tool), gollem.WithLoopLimit(5))
		_, err := agent.Execute(t.Context(), gollem.Text("search something"))
		gt.NoError(t, err)
		gt.Equal(t, 2, callCount)
		gt.False(t, toolRunCalled)
		gt.NotNil(t, receivedError)
		gt.S(t, receivedError.Error()).Contains("required")
	})

	t.Run("ToolSet tool args are validated", func(t *testing.T) {
		callCount := 0
		var receivedError error

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:   "call_1",
										Name: "toolset_search",
										Arguments: map[string]any{
											"query": 999, // wrong type
										},
									},
								},
							}, nil
						}
						if callCount == 2 {
							if len(input) > 0 {
								if funcResp, ok := input[0].(gollem.FunctionResponse); ok {
									receivedError = funcResp.Error
								}
							}
							return &gollem.Response{Texts: []string{"Done"}}, nil
						}
						return &gollem.Response{Texts: []string{"unexpected"}}, nil
					},
				}, nil
			},
		}

		toolSetRunCalled := false
		ts := &mockToolSet{
			specs: []gollem.ToolSpec{
				{
					Name: "toolset_search",
					Parameters: map[string]*gollem.Parameter{
						"query": {Type: gollem.TypeString, Required: true},
					},
				},
			},
			run: func(ctx context.Context, name string, args map[string]any) (map[string]any, error) {
				toolSetRunCalled = true
				return map[string]any{"result": "ok"}, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithToolSets(ts), gollem.WithLoopLimit(5))
		_, err := agent.Execute(t.Context(), gollem.Text("search"))
		gt.NoError(t, err)
		gt.Equal(t, 2, callCount)
		gt.False(t, toolSetRunCalled)
		gt.NotNil(t, receivedError)
		gt.S(t, receivedError.Error()).Contains("toolset_search")
	})

	t.Run("WithDisableArgsValidation skips validation", func(t *testing.T) {
		callCount := 0

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							// LLM sends wrong type, but validation is disabled
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:   "call_1",
										Name: "echo",
										Arguments: map[string]any{
											"message": "hello",
										},
									},
								},
							}, nil
						}
						return &gollem.Response{Texts: []string{"Done"}}, nil
					},
				}, nil
			},
		}

		toolRunCalled := false
		tool := &mockTool{
			spec: gollem.ToolSpec{
				Name: "echo",
				Parameters: map[string]*gollem.Parameter{
					"message": {Type: gollem.TypeInteger, Required: true}, // Spec says integer
				},
			},
			run: func(ctx context.Context, args map[string]any) (map[string]any, error) {
				toolRunCalled = true
				return map[string]any{"echo": args["message"]}, nil
			},
		}

		agent := gollem.New(mockClient,
			gollem.WithTools(tool),
			gollem.WithDisableArgsValidation(),
			gollem.WithLoopLimit(5),
		)
		_, err := agent.Execute(t.Context(), gollem.Text("echo something"))
		gt.NoError(t, err)
		gt.True(t, toolRunCalled) // tool.Run should be called despite type mismatch
	})
}

func TestDefaultStrategyWithExecuteResponse(t *testing.T) {
	t.Run("default strategy generates conclusion for LLM response without tool calls", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}

		// Mock session that returns a response without function calls
		mockSession := &mock.SessionMock{}
		mockSession.GenerateFunc = func(ctx context.Context, inputs []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			return &gollem.Response{
				Texts:         []string{"Task completed successfully"},
				FunctionCalls: []*gollem.FunctionCall{}, // No tool calls
			}, nil
		}

		mockClient.NewSessionFunc = func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return mockSession, nil
		}

		agent := gollem.New(mockClient) // Uses default strategy
		result, err := agent.Execute(context.Background(), gollem.Text("test task"))

		gt.NoError(t, err)
		gt.NotNil(t, result)
		gt.Equal(t, "Task completed successfully", result.String())
	})

	t.Run("default strategy continues with tool calls", func(t *testing.T) {
		mockClient := &mock.LLMClientMock{}

		callCount := 0
		mockSession := &mock.SessionMock{}
		mockSession.GenerateFunc = func(ctx context.Context, inputs []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
			callCount++
			if callCount == 1 {
				// First call: return tool call
				return &gollem.Response{
					Texts: []string{"Calling tool"},
					FunctionCalls: []*gollem.FunctionCall{
						{Name: "test_tool", ID: "call_1", Arguments: map[string]any{}},
					},
				}, nil
			} else {
				// Second call: return final response
				return &gollem.Response{
					Texts:         []string{"Tool execution completed"},
					FunctionCalls: []*gollem.FunctionCall{}, // No more tool calls
				}, nil
			}
		}

		mockClient.NewSessionFunc = func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return mockSession, nil
		}

		// Add a test tool
		testTool := &RandomNumberTool{}
		agent := gollem.New(mockClient, gollem.WithTools(testTool))
		result, err := agent.Execute(context.Background(), gollem.Text("test task"))

		gt.NoError(t, err)
		gt.NotNil(t, result)
		gt.Equal(t, "Tool execution completed", result.String())
		gt.Equal(t, 2, callCount)
	})
}

// mockHistoryRepository is a simple in-memory HistoryRepository for testing.
type mockHistoryRepository struct {
	loadFn func(ctx context.Context, sessionID string) (*gollem.History, error)
	saveFn func(ctx context.Context, sessionID string, history *gollem.History) error

	loadCalls []string
	saveCalls []*gollem.History
}

func (m *mockHistoryRepository) Load(ctx context.Context, sessionID string) (*gollem.History, error) {
	m.loadCalls = append(m.loadCalls, sessionID)
	if m.loadFn != nil {
		return m.loadFn(ctx, sessionID)
	}
	return nil, nil
}

func (m *mockHistoryRepository) Save(ctx context.Context, sessionID string, history *gollem.History) error {
	m.saveCalls = append(m.saveCalls, history)
	if m.saveFn != nil {
		return m.saveFn(ctx, sessionID, history)
	}
	return nil
}

func TestWithHistoryRepository(t *testing.T) {
	newSimpleSession := func() *mock.SessionMock {
		callCount := 0
		return &mock.SessionMock{
			GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
				callCount++
				if callCount == 1 {
					return &gollem.Response{Texts: []string{"done"}}, nil
				}
				return &gollem.Response{}, nil
			},
			HistoryFunc: func() (*gollem.History, error) {
				return &gollem.History{Version: gollem.HistoryVersion}, nil
			},
			AppendHistoryFunc: func(history *gollem.History) error { return nil },
		}
	}

	t.Run("Load is called once on first Execute, Save is called after GenerateContent", func(t *testing.T) {
		repo := &mockHistoryRepository{}
		mockSession := newSimpleSession()

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return mockSession, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithHistoryRepository(repo, "sess1"))
		_, err := agent.Execute(context.Background(), gollem.Text("hello"))
		gt.NoError(t, err)

		// Load should be called exactly once (on first Execute)
		gt.Equal(t, 1, len(repo.loadCalls))
		gt.Equal(t, "sess1", repo.loadCalls[0])

		// Save should be called at least once (after GenerateContent)
		gt.Equal(t, true, len(repo.saveCalls) > 0)
	})

	t.Run("Load is called only on first Execute, not on subsequent Executes", func(t *testing.T) {
		repo := &mockHistoryRepository{}
		mockSession := newSimpleSession()

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return mockSession, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithHistoryRepository(repo, "sess1"))
		_, err := agent.Execute(context.Background(), gollem.Text("first"))
		gt.NoError(t, err)
		_, err = agent.Execute(context.Background(), gollem.Text("second"))
		gt.NoError(t, err)

		// Load should only be called once (only when currentSession is nil)
		gt.Equal(t, 1, len(repo.loadCalls))
	})

	t.Run("WithHistory and WithHistoryRepository together returns error", func(t *testing.T) {
		repo := &mockHistoryRepository{}
		existingHistory := &gollem.History{Version: gollem.HistoryVersion}

		mockSession := newSimpleSession()
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return mockSession, nil
			},
		}

		agent := gollem.New(mockClient,
			gollem.WithHistory(existingHistory),
			gollem.WithHistoryRepository(repo, "sess1"),
		)
		_, err := agent.Execute(context.Background(), gollem.Text("hello"))
		gt.Error(t, err)
	})

	t.Run("Load error is propagated from Execute", func(t *testing.T) {
		loadErr := errors.New("load failed")
		repo := &mockHistoryRepository{
			loadFn: func(ctx context.Context, sessionID string) (*gollem.History, error) {
				return nil, loadErr
			},
		}

		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return newSimpleSession(), nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithHistoryRepository(repo, "sess1"))
		_, err := agent.Execute(context.Background(), gollem.Text("hello"))
		gt.Error(t, err)
	})

	t.Run("Save error stops the execution loop", func(t *testing.T) {
		saveErr := errors.New("save failed")
		repo := &mockHistoryRepository{
			saveFn: func(ctx context.Context, sessionID string, history *gollem.History) error {
				return saveErr
			},
		}

		mockSession := newSimpleSession()
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return mockSession, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithHistoryRepository(repo, "sess1"))
		_, err := agent.Execute(context.Background(), gollem.Text("hello"))
		gt.Error(t, err)
	})

	t.Run("Existing history is loaded from repository into session", func(t *testing.T) {
		savedHistory := &gollem.History{
			Version: gollem.HistoryVersion,
			Messages: []gollem.Message{
				{Role: gollem.RoleUser},
			},
		}
		repo := &mockHistoryRepository{
			loadFn: func(ctx context.Context, sessionID string) (*gollem.History, error) {
				return savedHistory, nil
			},
		}

		var sessionOptions []gollem.SessionOption
		mockSession := newSimpleSession()
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				sessionOptions = options
				return mockSession, nil
			},
		}

		agent := gollem.New(mockClient, gollem.WithHistoryRepository(repo, "sess1"))
		_, err := agent.Execute(context.Background(), gollem.Text("hello"))
		gt.NoError(t, err)

		// Verify that NewSession was called with the loaded history
		cfg := gollem.NewSessionConfig(sessionOptions...)
		gt.NotEqual(t, nil, cfg.History())
		gt.Equal(t, 1, len(cfg.History().Messages))
	})
}

func TestStackTraceWithAgentExecute(t *testing.T) {
	t.Run("agent_execute and tool_exec stack traces point to gollem internal code", func(t *testing.T) {
		callCount := 0
		mockClient := &mock.LLMClientMock{
			NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
				return &mock.SessionMock{
					GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
						callCount++
						if callCount == 1 {
							return &gollem.Response{
								FunctionCalls: []*gollem.FunctionCall{
									{
										ID:   "call_1",
										Name: "random_number",
										Arguments: map[string]any{
											"min": float64(1),
											"max": float64(10),
										},
									},
								},
							}, nil
						}
						return &gollem.Response{
							Texts: []string{"Done"},
						}, nil
					},
				}, nil
			},
		}

		rec := trace.New(trace.WithStackTrace())
		agent := gollem.New(mockClient,
			gollem.WithTools(&RandomNumberTool{}),
			gollem.WithLoopLimit(5),
			gollem.WithTrace(rec),
		)

		_, err := agent.Execute(t.Context(), gollem.Text("generate number"))
		gt.NoError(t, err)

		tr := rec.Trace()
		gt.Value(t, tr).NotNil()

		rootSpan := tr.RootSpan
		gt.V(t, rootSpan.Kind).Equal(trace.SpanKindAgentExecute)

		// agent_execute span: top frame must be gollem.go (the Execute method)
		gt.A(t, rootSpan.StackTrace).Longer(0)
		gt.S(t, rootSpan.StackTrace[0].File).Contains("gollem.go")
		gt.S(t, rootSpan.StackTrace[0].Function).Contains("Agent).Execute")
		gt.N(t, rootSpan.StackTrace[0].Line).Greater(0)

		// Find tool_exec span among children
		var toolSpan *trace.Span
		for _, child := range rootSpan.Children {
			if child.Kind == trace.SpanKindToolExec {
				toolSpan = child
				break
			}
		}
		gt.Value(t, toolSpan).NotNil()

		// tool_exec span: top frame must be gollem.go (the executeToolCall function)
		gt.A(t, toolSpan.StackTrace).Longer(0)
		gt.S(t, toolSpan.StackTrace[0].File).Contains("gollem.go")
		gt.S(t, toolSpan.StackTrace[0].Function).Contains("executeToolCall")
		gt.N(t, toolSpan.StackTrace[0].Line).Greater(0)
	})
}

// TestPromptCacheLive verifies prompt-cache behavior against the real APIs of
// every supported provider. The shared observable across all providers is a
// cache *read* (hit) on a repeated large prompt prefix; Claude additionally
// reports a cache *write* on the first call. Each provider is gated on its own
// TEST_ environment variables and skipped when unset.
func TestPromptCacheLive(t *testing.T) {
	// runCacheCheck sends a large prefix twice through one session and observes the
	// prompt-cache token accounting.
	//   - requireHit:      assert the second call reports a cache read. Use for
	//                      providers whose caching is deterministic for a repeated
	//                      prefix (Claude explicit control, OpenAI automatic). Not
	//                      set for Gemini, whose implicit caching is best-effort.
	//   - expectCreation:  assert the first (cold) call reports a cache write. Only
	//                      Claude distinguishes and reports cache writes.
	// The token-accounting invariants (InputToken is total input; cache counts are
	// non-negative) are always asserted, so the observation path is verified for
	// every provider even when no hit occurs.
	runCacheCheck := func(t *testing.T, client gollem.LLMClient, requireHit, expectCreation bool) {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		// Build a prefix comfortably above every model's minimum cacheable length
		// (up to 4096 tokens). Placed in the user input so it reaches providers
		// that do not send the system prompt as a message (OpenAI). A per-run
		// nonce keeps the prefix unique so the first call is a cold cache write
		// (not a hit on a cache left over from a previous run within the TTL).
		var sb strings.Builder
		fmt.Fprintf(&sb, "session-nonce-%d-%d ", os.Getpid(), time.Now().UnixNano())
		for i := 0; i < 1000; i++ {
			sb.WriteString("Context paragraph for prompt caching validation. ")
		}
		bigPrefix := sb.String()

		session, err := client.NewSession(ctx, gollem.WithSessionPromptCache(true))
		gt.NoError(t, err).Required()

		first, err := session.Generate(ctx,
			[]gollem.Input{gollem.Text(bigPrefix), gollem.Text("Reply with the single word: one")},
			gollem.WithMaxTokens(2048))
		gt.NoError(t, err).Required()

		second, err := session.Generate(ctx,
			[]gollem.Input{gollem.Text("Reply with the single word: two")},
			gollem.WithMaxTokens(2048))
		gt.NoError(t, err).Required()

		t.Logf("first : input=%d creation=%d read=%d",
			first.InputToken, first.CacheCreationInputToken, first.CacheReadInputToken)
		t.Logf("second: input=%d creation=%d read=%d",
			second.InputToken, second.CacheCreationInputToken, second.CacheReadInputToken)

		// Accounting invariants (always hold, cache hit or not).
		gt.Value(t, second.InputToken >= second.CacheReadInputToken).Equal(true)
		gt.Value(t, second.CacheReadInputToken >= 0).Equal(true)
		gt.Value(t, first.CacheCreationInputToken >= 0).Equal(true)

		if requireHit {
			// The repeated prefix was served from the cache on the second call.
			gt.Value(t, second.CacheReadInputToken > 0).Equal(true)
		}
		if expectCreation {
			// The cold first call wrote the prefix to the cache.
			gt.Value(t, first.CacheCreationInputToken > 0).Equal(true)
		}
	}

	t.Run("claude", func(t *testing.T) {
		apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
		if !ok {
			t.Skip("TEST_CLAUDE_API_KEY is not set")
		}
		client, err := claude.New(context.Background(), apiKey)
		gt.NoError(t, err).Required()
		// Claude has explicit cache control: both write and read are deterministic.
		runCacheCheck(t, client, true, true)
	})

	t.Run("openai", func(t *testing.T) {
		apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
		if !ok {
			t.Skip("TEST_OPENAI_API_KEY is not set")
		}
		client, err := openai.New(context.Background(), apiKey)
		gt.NoError(t, err).Required()
		// OpenAI caches automatically for long repeated prefixes and reports reads
		// only (no creation count).
		runCacheCheck(t, client, true, false)
	})

	t.Run("gemini", func(t *testing.T) {
		projectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
		if !ok {
			t.Skip("TEST_GCP_PROJECT_ID is not set")
		}
		location, ok := os.LookupEnv("TEST_GCP_LOCATION")
		if !ok {
			t.Skip("TEST_GCP_LOCATION is not set")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		defer cancel()

		// Gemini caches implicitly (no client-side control). Implicit caching is
		// reliable on gemini-2.5-flash for a large repeated prefix but flaky or
		// absent on other models, so pin the model here to keep the assertion
		// deterministic. 2.5 models need an explicit thinking budget instead of
		// the default thinking level.
		client, err := gemini.New(ctx, projectID, location,
			gemini.WithModel("gemini-2.5-flash"), gemini.WithThinkingBudget(0))
		gt.NoError(t, err).Required()

		var sb strings.Builder
		for i := 0; i < 2000; i++ {
			sb.WriteString("Context paragraph for prompt caching validation. ")
		}
		input := []gollem.Input{gollem.Text(sb.String()), gollem.Text("Reply with the single word: one")}

		// First request primes the implicit cache; a fresh session with identical
		// input then hits it.
		prime, err := client.NewSession(ctx)
		gt.NoError(t, err).Required()
		_, err = prime.Generate(ctx, input, gollem.WithMaxTokens(64))
		gt.NoError(t, err).Required()

		hit, err := client.NewSession(ctx)
		gt.NoError(t, err).Required()
		resp, err := hit.Generate(ctx, input, gollem.WithMaxTokens(64))
		gt.NoError(t, err).Required()

		t.Logf("gemini implicit cache: input=%d read=%d", resp.InputToken, resp.CacheReadInputToken)
		gt.Value(t, resp.CacheReadInputToken > 0).Equal(true)
		gt.Value(t, resp.InputToken >= resp.CacheReadInputToken).Equal(true)
	})
}

// captureStrategy records the LastResponse the agent produced, for asserting the
// streaming token accumulation.
type captureStrategy struct {
	inputs []gollem.Input
	got    *gollem.Response
}

func (s *captureStrategy) Init(ctx context.Context, inputs []gollem.Input) error {
	s.inputs = inputs
	return nil
}

func (s *captureStrategy) Handle(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	if state.LastResponse != nil {
		s.got = state.LastResponse
		return nil, &gollem.ExecuteResponse{Texts: []string{"done"}}, nil
	}
	return s.inputs, nil, nil
}

func (s *captureStrategy) Tools(ctx context.Context) ([]gollem.Tool, error) { return nil, nil }

func TestStreamingUsageNotMultiplied(t *testing.T) {
	// A provider emits the per-call usage snapshot on every chunk (the running
	// total, not a delta). The agent must report the single total, not the sum
	// over chunks.
	mockClient := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return &mock.SessionMock{
				StreamFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
					ch := make(chan *gollem.Response)
					go func() {
						defer close(ch)
						for i := 0; i < 3; i++ {
							ch <- &gollem.Response{
								Texts:               []string{"chunk"},
								InputToken:          100,
								OutputToken:         5,
								CacheReadInputToken: 50,
							}
						}
					}()
					return ch, nil
				},
			}, nil
		},
	}

	strat := &captureStrategy{}
	s := gollem.New(mockClient,
		gollem.WithResponseMode(gollem.ResponseModeStreaming),
		gollem.WithStrategy(strat),
		gollem.WithLoopLimit(3),
	)
	_, err := s.Execute(t.Context(), gollem.Text("hi"))
	gt.NoError(t, err)
	gt.Value(t, strat.got).NotNil().Required()

	// Three chunks each carrying 100/5/50 must not become 300/15/150.
	gt.Equal(t, 100, strat.got.InputToken)
	gt.Equal(t, 5, strat.got.OutputToken)
	gt.Equal(t, 50, strat.got.CacheReadInputToken)
}

func TestStreamingAccumulatesThoughts(t *testing.T) {
	// Reasoning text arrives in chunks like any other content, so the response
	// handed to the strategy must carry it as the blocking mode does.
	mockClient := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return &mock.SessionMock{
				StreamFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (<-chan *gollem.Response, error) {
					ch := make(chan *gollem.Response)
					go func() {
						defer close(ch)
						ch <- &gollem.Response{Thoughts: []string{"first thought"}}
						ch <- &gollem.Response{Thoughts: []string{"second thought"}}
						ch <- &gollem.Response{Texts: []string{"answer"}}
					}()
					return ch, nil
				},
			}, nil
		},
	}

	strat := &captureStrategy{}
	s := gollem.New(mockClient,
		gollem.WithResponseMode(gollem.ResponseModeStreaming),
		gollem.WithStrategy(strat),
		gollem.WithLoopLimit(3),
	)
	_, err := s.Execute(t.Context(), gollem.Text("hi"))
	gt.NoError(t, err)
	gt.Value(t, strat.got).NotNil().Required()

	gt.A(t, strat.got.Thoughts).Length(2).Required()
	gt.Equal(t, "first thought", strat.got.Thoughts[0])
	gt.Equal(t, "second thought", strat.got.Thoughts[1])
}
