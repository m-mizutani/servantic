package claude

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// apiClient is the interface for Claude API calls (unexported for encapsulation)
type apiClient interface {
	MessagesNew(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error)
	MessagesNewStreaming(ctx context.Context, params anthropic.MessageNewParams) *ssestream.Stream[anthropic.MessageStreamEventUnion]
	MessagesCountTokens(ctx context.Context, params anthropic.MessageCountTokensParams) (*anthropic.MessageTokensCount, error)
}

// realAPIClient wraps the actual Claude client
type realAPIClient struct {
	client *anthropic.Client
}

func (r *realAPIClient) MessagesNew(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
	return r.client.Messages.New(ctx, params, option.WithRequestTimeout(defaultNonStreamingTimeout))
}

func (r *realAPIClient) MessagesNewStreaming(ctx context.Context, params anthropic.MessageNewParams) *ssestream.Stream[anthropic.MessageStreamEventUnion] {
	return r.client.Messages.NewStreaming(ctx, params)
}

func (r *realAPIClient) MessagesCountTokens(ctx context.Context, params anthropic.MessageCountTokensParams) (*anthropic.MessageTokensCount, error) {
	return r.client.Messages.CountTokens(ctx, params)
}
