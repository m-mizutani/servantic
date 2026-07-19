package gollem_test

import (
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/m-mizutani/gt"
)

func TestSessionPromptCacheOption(t *testing.T) {
	t.Run("disabled by default", func(t *testing.T) {
		cfg := gollem.NewSessionConfig()
		gt.False(t, cfg.PromptCache())
	})

	t.Run("enabled via WithSessionPromptCache", func(t *testing.T) {
		cfg := gollem.NewSessionConfig(gollem.WithSessionPromptCache(true))
		gt.True(t, cfg.PromptCache())
	})

	t.Run("explicitly disabled", func(t *testing.T) {
		cfg := gollem.NewSessionConfig(gollem.WithSessionPromptCache(false))
		gt.False(t, cfg.PromptCache())
	})
}
