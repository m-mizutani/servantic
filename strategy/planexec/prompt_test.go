package planexec_test

import (
	"context"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/strategy/planexec"
	"github.com/m-mizutani/gt"
)

type multiParamTool struct{}

func (t *multiParamTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "multi_param_tool",
		Description: "A tool with several parameters",
		Parameters: map[string]*gollem.Parameter{
			"zulu":  {Type: gollem.TypeString},
			"alpha": {Type: gollem.TypeString},
			"mike":  {Type: gollem.TypeString},
			"bravo": {Type: gollem.TypeString},
		},
	}
}

func (t *multiParamTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return nil, nil
}

// TestBuildToolList pins the tool list to be identical between calls. It is
// embedded in the plan and execute prompts, so a parameter order taken from Go
// map iteration would change the prompt text between otherwise identical runs.
func TestBuildToolList(t *testing.T) {
	tools := []gollem.Tool{&multiParamTool{}}

	first := planexec.BuildToolList(tools)
	gt.S(t, first).Contains("Parameters: alpha, bravo, mike, zulu")

	for i := 0; i < 100; i++ {
		gt.Equal(t, first, planexec.BuildToolList(tools))
	}
}
