// Package react implements the ReAct (Reasoning and Acting) strategy for gollem.
//
// ReAct is a framework that combines reasoning and acting in language models.
// It alternates between thought (reasoning), action (tool use), and observation (results)
// to solve complex problems step by step.
//
// Basic usage:
//
//	strategy := react.New(llmClient,
//	    react.WithMaxIterations(20),
//	    react.WithMaxRepeatedActions(3),
//	)
//
//	agent := gollem.New(llmClient, gollem.WithStrategy(strategy))
//	response, err := agent.Execute(ctx, gollem.Text("What is the weather in Tokyo?"))
package react

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
)

const (
	// DefaultMaxIterations is the default maximum number of iterations
	DefaultMaxIterations = 20
	// DefaultMaxRepeatedActions is the default maximum number of repeated actions
	DefaultMaxRepeatedActions = 3
	// MaxConsecutiveErrors is the maximum number of consecutive errors before giving up
	MaxConsecutiveErrors = 3
)

// New creates a new ReAct strategy instance
func New(client gollem.LLMClient, options ...Option) *Strategy {
	s := &Strategy{
		llm:                client,
		maxIterations:      DefaultMaxIterations,
		maxRepeatedActions: DefaultMaxRepeatedActions,
		trace:              make([]TAOEntry, 0),
		actionHistory:      make([]string, 0),
		repeatedCount:      make(map[string]int),
	}

	for _, opt := range options {
		opt(s)
	}

	return s
}

// Init initializes the strategy with initial inputs
func (s *Strategy) Init(ctx context.Context, inputs []gollem.Input) error {
	// Reset all state
	s.trace = make([]TAOEntry, 0)
	s.currentEntry = nil
	s.actionHistory = make([]string, 0)
	s.repeatedCount = make(map[string]int)
	s.consecutiveErrors = 0
	s.startTime = time.Now()
	s.endTime = time.Time{}

	return nil
}

// Tools returns the tools provided by this strategy (none for ReAct)
func (s *Strategy) Tools(ctx context.Context) ([]gollem.Tool, error) {
	return []gollem.Tool{}, nil
}

// Handle implements the ReAct loop logic
func (s *Strategy) Handle(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	// Phase 0: Initialize on first iteration
	if state.Iteration == 0 {
		return s.handleInitialization(state)
	}

	// Safety check: max iterations
	if state.Iteration >= s.maxIterations {
		s.endTime = time.Now()
		return nil, &gollem.ExecuteResponse{
			Texts: []string{fmt.Sprintf("Maximum iterations (%d) reached without completion", s.maxIterations)},
		}, nil
	}

	// Phase 1-2: Process LLM response (Thought + Action)
	if state.LastResponse != nil {
		return s.handleThoughtAndAction(ctx, state)
	}

	// Phase 3: Process tool results (Observation)
	if len(state.NextInput) > 0 {
		return s.handleObservation(ctx, state)
	}

	// Fallback: continue
	return state.NextInput, nil, nil
}

// handleInitialization handles the first iteration (Phase 0)
func (s *Strategy) handleInitialization(state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	// Create initial TAO entry
	s.addTAOEntry(0)

	// Build inputs with system prompt and thought prompt
	systemPrompt := s.systemPrompt
	if systemPrompt == "" {
		systemPrompt = DefaultSystemPrompt
	}

	inputs := []gollem.Input{
		gollem.Text(systemPrompt + "\n\n" + s.buildThoughtPrompt()),
	}
	inputs = append(inputs, state.InitInput...)

	return inputs, nil, nil
}

// handleThoughtAndAction handles Thought and Action phases
func (s *Strategy) handleThoughtAndAction(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	resp := state.LastResponse

	// Record thought
	if len(resp.Texts) > 0 {
		s.recordThought(strings.Join(resp.Texts, " "))

		// Trace event: thought
		if rec := trace.HandlerFrom(ctx); rec != nil {
			rec.AddEvent(ctx, "thought", &ThoughtEvent{
				Iteration: state.Iteration,
				Content:   strings.Join(resp.Texts, " "),
			})
		}
	}

	// Check if this is a final response (no tool calls)
	if len(resp.FunctionCalls) == 0 {
		// Record action as respond
		s.recordAction(ActionTypeRespond, nil, strings.Join(resp.Texts, " "))
		// Mark observation as complete (no tools executed)
		s.recordObservation(nil, true, nil)

		s.endTime = time.Now()
		return nil, &gollem.ExecuteResponse{
			Texts: resp.Texts,
		}, nil
	}

	// Record action as tool calls
	s.recordAction(ActionTypeToolCall, resp.FunctionCalls, "")

	// Trace event: action
	if rec := trace.HandlerFrom(ctx); rec != nil {
		var toolNames []string
		for _, fc := range resp.FunctionCalls {
			toolNames = append(toolNames, fc.Name)
		}
		rec.AddEvent(ctx, "action", &ActionEvent{
			Iteration:  state.Iteration,
			ActionType: string(ActionTypeToolCall),
			ToolNames:  toolNames,
		})
	}

	// Check for loops
	actionKey := s.generateActionKey(resp.FunctionCalls)
	if s.detectLoop(actionKey) {
		s.endTime = time.Now()
		return nil, &gollem.ExecuteResponse{
			Texts: []string{fmt.Sprintf("Loop detected: same action repeated %d times", s.maxRepeatedActions)},
		}, nil
	}

	// Continue to observation phase (tool execution handled by gollem core)
	return state.NextInput, nil, nil
}

// handleObservation handles the Observation phase
func (s *Strategy) handleObservation(ctx context.Context, state *gollem.StrategyState) ([]gollem.Input, *gollem.ExecuteResponse, error) {
	// Convert function responses to tool results
	toolResults, err := convertFunctionResponsesToToolResults(state.NextInput)
	if err != nil {
		return nil, nil, err
	}

	// Check for errors
	hasError := false
	var toolErr error
	for _, result := range toolResults {
		if !result.Success {
			hasError = true
			toolErr = goerr.Wrap(fmt.Errorf("tool execution failed: %s", result.Error), fmt.Sprintf("tool %s error", result.ToolName))
			break
		}
	}

	// Record observation
	s.recordObservation(toolResults, !hasError, toolErr)

	// Trace event: observation
	if rec := trace.HandlerFrom(ctx); rec != nil {
		rec.AddEvent(ctx, "observation", &ObservationEvent{
			Iteration: state.Iteration,
			Success:   !hasError,
		})
	}

	// Track consecutive errors
	if hasError {
		s.consecutiveErrors++
		if s.consecutiveErrors >= MaxConsecutiveErrors {
			s.endTime = time.Now()
			return nil, &gollem.ExecuteResponse{
				Texts: []string{fmt.Sprintf("Maximum consecutive errors (%d) reached", MaxConsecutiveErrors)},
			}, nil
		}
	} else {
		s.consecutiveErrors = 0
	}

	// Build observation prompt
	observationPrompt := gollem.Text(s.buildObservationPrompt(toolResults))

	// Create new TAO entry for next iteration
	s.addTAOEntry(state.Iteration + 1)

	// Return only observation prompt (state.NextInput contains raw FunctionResponse objects
	// which would duplicate the information already formatted in observationPrompt)
	return []gollem.Input{observationPrompt}, nil, nil
}

// generateActionKey generates a unique key for an action (for loop detection)
func (s *Strategy) generateActionKey(calls []*gollem.FunctionCall) string {
	if len(calls) == 0 {
		return "no_action"
	}

	var parts []string
	for _, call := range calls {
		parts = append(parts, call.Name)
	}
	return strings.Join(parts, ",")
}

// detectLoop detects if the same action is being repeated
func (s *Strategy) detectLoop(actionKey string) bool {
	s.actionHistory = append(s.actionHistory, actionKey)
	s.repeatedCount[actionKey]++

	return s.repeatedCount[actionKey] >= s.maxRepeatedActions
}
