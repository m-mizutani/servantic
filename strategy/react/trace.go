package react

import (
	"encoding/json"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

// addTAOEntry adds a new TAO entry to the trace
func (s *Strategy) addTAOEntry(iteration int) {
	entry := &TAOEntry{
		Iteration: iteration,
		Timestamp: time.Now(),
	}
	s.currentEntry = entry
}

// recordThought records the thought phase data
func (s *Strategy) recordThought(content string) {
	if s.currentEntry == nil {
		return
	}
	s.currentEntry.Thought = &ThoughtData{
		Content:   content,
		Reasoning: content,
	}
}

// recordAction records the action phase data
func (s *Strategy) recordAction(actionType ActionType, toolCalls []*gollem.FunctionCall, response string) {
	if s.currentEntry == nil {
		return
	}
	s.currentEntry.Action = &ActionData{
		Type:      actionType,
		ToolCalls: toolCalls,
		Response:  response,
	}
}

// recordObservation records the observation phase data
func (s *Strategy) recordObservation(toolResults []ToolResult, success bool, err error) {
	if s.currentEntry == nil {
		return
	}

	var errStr string
	if err != nil {
		errStr = err.Error()
	}

	s.currentEntry.Observation = &ObservationData{
		ToolResults: toolResults,
		Success:     success,
		Error:       errStr,
	}

	// Finalize this entry and add to trace
	s.trace = append(s.trace, *s.currentEntry)
	s.currentEntry = nil
}

// convertFunctionResponsesToToolResults converts gollem.FunctionResponse to ToolResult.
// An encoding failure is returned rather than ignored: Output feeds the observation prompt,
// so dropping it would leave the model reasoning about an empty tool result.
func convertFunctionResponsesToToolResults(inputs []gollem.Input) ([]ToolResult, error) {
	var results []ToolResult

	for _, input := range inputs {
		if fr, ok := input.(gollem.FunctionResponse); ok {
			result := ToolResult{
				ToolName: fr.Name,
				Success:  fr.Error == nil,
			}

			if fr.Error != nil {
				result.Error = fr.Error.Error()
			} else {
				// Convert Data to string representation
				dataBytes, err := json.Marshal(fr.Data)
				if err != nil {
					return nil, goerr.Wrap(err, "failed to encode tool result for observation",
						goerr.V("tool", fr.Name))
				}
				result.Output = string(dataBytes)
			}

			results = append(results, result)
		}
	}

	return results, nil
}

// ExportTrace exports the complete trace data
func (s *Strategy) ExportTrace() *TraceExport {
	summary := s.calculateSummary()
	metadata := s.buildMetadata()

	return &TraceExport{
		Entries:  s.trace,
		Summary:  summary,
		Metadata: metadata,
	}
}

// ExportTraceJSON exports the trace data as JSON
func (s *Strategy) ExportTraceJSON() ([]byte, error) {
	export := s.ExportTrace()
	return json.MarshalIndent(export, "", "  ")
}

// calculateSummary calculates summary statistics from the trace
func (s *Strategy) calculateSummary() TraceSummary {
	totalIterations := len(s.trace)
	toolCallsCount := 0
	successCount := 0

	for _, entry := range s.trace {
		if entry.Action != nil && len(entry.Action.ToolCalls) > 0 {
			toolCallsCount += len(entry.Action.ToolCalls)
		}
		if entry.Observation != nil && entry.Observation.Success {
			successCount++
		}
	}

	successRate := 0.0
	if toolCallsCount > 0 {
		successRate = float64(successCount) / float64(totalIterations)
	}

	duration := time.Duration(0)
	if !s.endTime.IsZero() && !s.startTime.IsZero() {
		duration = s.endTime.Sub(s.startTime)
	}

	return TraceSummary{
		TotalIterations: totalIterations,
		ToolCallsCount:  toolCallsCount,
		SuccessRate:     successRate,
		Duration:        duration,
	}
}

// buildMetadata builds metadata for the trace
func (s *Strategy) buildMetadata() TraceMetadata {
	completionType := "success"
	if len(s.trace) >= s.maxIterations {
		completionType = "max_iterations"
	}

	return TraceMetadata{
		Strategy:       "react",
		StartTime:      s.startTime,
		EndTime:        s.endTime,
		CompletionType: completionType,
	}
}
