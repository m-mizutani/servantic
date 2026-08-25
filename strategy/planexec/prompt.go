package planexec

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"slices"
	"strings"
	"text/template"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/goerr/v2"
)

//go:embed prompts/plan.md
var planPromptTemplate string

//go:embed prompts/execute.md
var executePromptTemplate string

//go:embed prompts/reflect.md
var reflectPromptTemplate string

//go:embed prompts/conclusion.md
var conclusionPromptTemplate string

// Pre-parsed templates for better performance
var (
	planTemplate       = template.Must(template.New("plan").Parse(planPromptTemplate))
	executeTemplate    = template.Must(template.New("execute").Parse(executePromptTemplate))
	reflectTemplate    = template.Must(template.New("reflect").Parse(reflectPromptTemplate))
	conclusionTemplate = template.Must(template.New("conclusion").Parse(conclusionPromptTemplate))
)

// buildPlanPrompt creates a prompt for analyzing and planning
func buildPlanPrompt(_ context.Context, inputs []gollem.Input, tools []gollem.Tool) []gollem.Input {
	// Combine all input texts
	var inputTexts []string
	for _, input := range inputs {
		if text, ok := input.(gollem.Text); ok {
			inputTexts = append(inputTexts, string(text))
		}
	}

	userRequest := strings.Join(inputTexts, " ")

	// Build tool list
	toolList := buildToolList(tools)

	var buf bytes.Buffer
	if err := planTemplate.Execute(&buf, map[string]interface{}{
		"UserRequest": userRequest,
		"ToolList":    toolList,
	}); err != nil {
		panic(goerr.Wrap(err, "failed to execute plan template"))
	}

	return []gollem.Input{gollem.Text(buf.String())}
}

// buildExecutePrompt creates a prompt for executing a specific task
func buildExecutePrompt(ctx context.Context, task *Task, plan *Plan, currentIteration, maxIterations int) []gollem.Input {
	// Build list of completed tasks
	var completedTasks []string
	completedTaskCount := 0
	for _, t := range plan.Tasks {
		if t.State == TaskStateCompleted {
			completedTaskCount++
			completedTasks = append(completedTasks, fmt.Sprintf("[ID: %s] %s", t.ID, t.Description))
			if t.Result != "" {
				completedTasks = append(completedTasks, fmt.Sprintf("   Result: %s", t.Result))
			}
		}
	}

	completedStr := "None"
	if len(completedTasks) > 0 {
		completedStr = strings.Join(completedTasks, "\n")
	}

	remainingIterations := maxIterations - currentIteration

	var buf bytes.Buffer
	if err := executeTemplate.Execute(&buf, map[string]interface{}{
		"Goal":                plan.Goal,
		"ContextSummary":      plan.ContextSummary,
		"Constraints":         plan.Constraints,
		"TaskDescription":     task.Description,
		"CompletedTasks":      completedStr,
		"CurrentIteration":    currentIteration,
		"MaxIterations":       maxIterations,
		"CompletedTaskCount":  completedTaskCount,
		"RemainingIterations": remainingIterations,
	}); err != nil {
		panic(goerr.Wrap(err, "failed to execute execute template"))
	}

	return []gollem.Input{gollem.Text(buf.String())}
}

// buildReflectPrompt creates a prompt for reflection after task completion
func buildReflectPrompt(ctx context.Context, plan *Plan, latestResult string, tools []gollem.Tool, currentIteration, maxIterations int) []gollem.Input {
	// Build completed tasks list
	var completedTasks []string
	var remainingTasks []string
	completedTaskCount := 0

	for _, task := range plan.Tasks {
		taskStr := fmt.Sprintf("[ID: %s] %s", task.ID, task.Description)

		switch task.State {
		case TaskStateCompleted:
			completedTaskCount++
			completedTasks = append(completedTasks, taskStr)
		case TaskStatePending:
			remainingTasks = append(remainingTasks, taskStr)
		}
	}

	completedStr := strings.Join(completedTasks, "\n")
	if completedStr == "" {
		completedStr = "None"
	}

	remainingStr := strings.Join(remainingTasks, "\n")
	if remainingStr == "" {
		remainingStr = "None"
	}

	remainingIterations := maxIterations - currentIteration

	// Build tool list
	toolList := buildToolList(tools)

	var buf bytes.Buffer
	if err := reflectTemplate.Execute(&buf, map[string]interface{}{
		"UserIntent":          plan.UserIntent,
		"Goal":                plan.Goal,
		"ContextSummary":      plan.ContextSummary, // Embedded context from planning phase
		"Constraints":         plan.Constraints,    // Embedded constraints from planning phase
		"CompletedTasks":      completedStr,
		"RemainingTasks":      remainingStr,
		"LatestResult":        latestResult,
		"ToolList":            toolList,
		"CurrentIteration":    currentIteration,
		"MaxIterations":       maxIterations,
		"CompletedTaskCount":  completedTaskCount,
		"RemainingIterations": remainingIterations,
	}); err != nil {
		panic(goerr.Wrap(err, "failed to execute reflect template"))
	}

	return []gollem.Input{gollem.Text(buf.String())}
}

// buildToolList creates a formatted list of available tools
func buildToolList(tools []gollem.Tool) string {
	if len(tools) == 0 {
		return "No tools available"
	}

	var toolDescriptions []string
	for _, tool := range tools {
		spec := tool.Spec()
		toolDesc := fmt.Sprintf("- **%s**: %s", spec.Name, spec.Description)

		// Add parameter information if available
		if len(spec.Parameters) > 0 {
			var params []string
			for paramName := range spec.Parameters {
				params = append(params, paramName)
			}
			// Sort so the prompt text is identical between identical executions
			slices.Sort(params)
			if len(params) > 0 {
				toolDesc += fmt.Sprintf("\n  Parameters: %s", strings.Join(params, ", "))
			}
		}

		toolDescriptions = append(toolDescriptions, toolDesc)
	}

	return strings.Join(toolDescriptions, "\n")
}

// buildConclusionPrompt creates a prompt for generating the final conclusion
func buildConclusionPrompt(plan *Plan, taskSummaries []string) string {
	var buf bytes.Buffer
	if err := conclusionTemplate.Execute(&buf, map[string]interface{}{
		"UserQuestion":   plan.UserQuestion,
		"UserIntent":     plan.UserIntent,
		"Goal":           plan.Goal,
		"CompletedTasks": strings.Join(taskSummaries, "\n"),
	}); err != nil {
		panic(goerr.Wrap(err, "failed to execute conclusion template"))
	}

	return buf.String()
}
