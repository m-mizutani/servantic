package otel

import "go.opentelemetry.io/otel/attribute"

// Attribute keys following OpenTelemetry semantic conventions where applicable.
func llmModelAttr(model string) attribute.KeyValue {
	return attribute.String("llm.model", model)
}

func llmInputTokensAttr(tokens int) attribute.KeyValue {
	return attribute.Int("llm.input_tokens", tokens)
}

func llmOutputTokensAttr(tokens int) attribute.KeyValue {
	return attribute.Int("llm.output_tokens", tokens)
}

func llmCacheCreationInputTokensAttr(tokens int) attribute.KeyValue {
	return attribute.Int("llm.cache_creation_input_tokens", tokens)
}

func llmCacheReadInputTokensAttr(tokens int) attribute.KeyValue {
	return attribute.Int("llm.cache_read_input_tokens", tokens)
}

func toolNameAttr(name string) attribute.KeyValue {
	return attribute.String("tool.name", name)
}

func toolArgsAttr(args string) attribute.KeyValue {
	return attribute.String("tool.args", args)
}

func eventDataAttr(data string) attribute.KeyValue {
	return attribute.String("event.data", data)
}
