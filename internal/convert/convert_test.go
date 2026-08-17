package convert_test

import (
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/convert"
	"github.com/m-mizutani/gt"
)

func TestMergeSystemIntoFirstUser(t *testing.T) {
	newText := func(t *testing.T, text string) gollem.MessageContent {
		t.Helper()
		c, err := gollem.NewTextContent(text)
		gt.NoError(t, err)
		return c
	}
	textOf := func(t *testing.T, c gollem.MessageContent) string {
		t.Helper()
		text, err := c.GetTextContent()
		gt.NoError(t, err)
		return text.Text
	}

	t.Run("prepends the system text to the first user message", func(t *testing.T) {
		messages := []gollem.Message{
			{Role: gollem.RoleSystem, Contents: []gollem.MessageContent{newText(t, "be brief")}},
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{newText(t, "hello")}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{newText(t, "hi")}},
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{newText(t, "again")}},
		}

		merged := convert.MergeSystemIntoFirstUser(messages)

		gt.Equal(t, 3, len(merged))
		gt.Equal(t, gollem.RoleUser, merged[0].Role)
		gt.Equal(t, 2, len(merged[0].Contents))
		gt.Equal(t, "be brief\n\n", textOf(t, merged[0].Contents[0]))
		gt.Equal(t, "hello", textOf(t, merged[0].Contents[1]))
		// Only the first user message receives the system text
		gt.Equal(t, 1, len(merged[2].Contents))
	})

	t.Run("does not modify the given messages", func(t *testing.T) {
		messages := []gollem.Message{
			{Role: gollem.RoleSystem, Contents: []gollem.MessageContent{newText(t, "be brief")}},
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{newText(t, "hello")}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{newText(t, "hi")}},
		}

		convert.MergeSystemIntoFirstUser(messages)

		gt.Equal(t, 3, len(messages))
		gt.Equal(t, gollem.RoleSystem, messages[0].Role)
		gt.Equal(t, gollem.RoleUser, messages[1].Role)
		gt.Equal(t, 1, len(messages[1].Contents))
		gt.Equal(t, "hello", textOf(t, messages[1].Contents[0]))
		gt.Equal(t, gollem.RoleAssistant, messages[2].Role)
	})

	t.Run("returns the input unchanged when there is no system message", func(t *testing.T) {
		messages := []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{newText(t, "hello")}},
		}

		gt.Equal(t, messages, convert.MergeSystemIntoFirstUser(messages))
	})

	t.Run("drops a system message that carries no text", func(t *testing.T) {
		call, err := gollem.NewToolCallContent("c1", "alpha", map[string]any{"x": 1})
		gt.NoError(t, err)
		messages := []gollem.Message{
			{Role: gollem.RoleSystem, Contents: []gollem.MessageContent{call}},
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{newText(t, "hello")}},
		}

		merged := convert.MergeSystemIntoFirstUser(messages)

		gt.Equal(t, 1, len(merged))
		gt.Equal(t, gollem.RoleUser, merged[0].Role)
		gt.Equal(t, 1, len(merged[0].Contents))
	})
}

func TestMergeConsecutiveToolMessages(t *testing.T) {
	newText := func(t *testing.T, text string) gollem.MessageContent {
		t.Helper()
		c, err := gollem.NewTextContent(text)
		gt.NoError(t, err)
		return c
	}
	newToolResponse := func(t *testing.T, id, name string) gollem.MessageContent {
		t.Helper()
		c, err := gollem.NewToolResponseContent(id, name, map[string]any{"ok": true}, false)
		gt.NoError(t, err)
		return c
	}

	t.Run("merges a run of consecutive tool messages into one", func(t *testing.T) {
		resp1 := newToolResponse(t, "c1", "alpha")
		resp2 := newToolResponse(t, "c2", "beta")
		messages := []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{newText(t, "go")}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{newText(t, "calling")}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp1}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp2}},
		}

		merged := convert.MergeConsecutiveToolMessages(messages)

		gt.Equal(t, 3, len(merged))
		gt.Equal(t, gollem.RoleTool, merged[2].Role)
		gt.Equal(t, []gollem.MessageContent{resp1, resp2}, merged[2].Contents)
	})

	t.Run("keeps tool messages that answer different call turns separate", func(t *testing.T) {
		messages := []gollem.Message{
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{newText(t, "first call")}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{newToolResponse(t, "c1", "alpha")}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{newText(t, "second call")}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{newToolResponse(t, "c2", "beta")}},
		}

		merged := convert.MergeConsecutiveToolMessages(messages)

		gt.Equal(t, messages, merged)
	})

	t.Run("merges three or more consecutive tool messages", func(t *testing.T) {
		messages := []gollem.Message{
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{newToolResponse(t, "c1", "alpha")}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{newToolResponse(t, "c2", "beta")}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{newToolResponse(t, "c3", "gamma")}},
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{newText(t, "next")}},
		}

		merged := convert.MergeConsecutiveToolMessages(messages)

		gt.Equal(t, 2, len(merged))
		gt.Equal(t, 3, len(merged[0].Contents))
		gt.Equal(t, gollem.RoleUser, merged[1].Role)
	})

	t.Run("keeps the Name and Metadata of the first message of a run", func(t *testing.T) {
		messages := []gollem.Message{
			{
				Role:     gollem.RoleTool,
				Contents: []gollem.MessageContent{newToolResponse(t, "c1", "alpha")},
				Name:     "alpha",
				Metadata: map[string]any{"source": "first"},
			},
			{
				Role:     gollem.RoleTool,
				Contents: []gollem.MessageContent{newToolResponse(t, "c2", "beta")},
				Name:     "beta",
				Metadata: map[string]any{"source": "second"},
			},
		}

		merged := convert.MergeConsecutiveToolMessages(messages)

		gt.Equal(t, 1, len(merged))
		gt.Equal(t, "alpha", merged[0].Name)
		gt.Equal(t, map[string]any{"source": "first"}, merged[0].Metadata)
	})

	t.Run("does not modify the given messages", func(t *testing.T) {
		messages := []gollem.Message{
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{newToolResponse(t, "c1", "alpha")}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{newToolResponse(t, "c2", "beta")}},
		}

		convert.MergeConsecutiveToolMessages(messages)

		gt.Equal(t, 2, len(messages))
		gt.Equal(t, 1, len(messages[0].Contents))
		gt.Equal(t, 1, len(messages[1].Contents))
	})

	t.Run("returns empty input unchanged", func(t *testing.T) {
		gt.Equal(t, 0, len(convert.MergeConsecutiveToolMessages(nil)))
		gt.Equal(t, 0, len(convert.MergeConsecutiveToolMessages([]gollem.Message{})))
	})
}
