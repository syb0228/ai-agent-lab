"""Helpers for adapting legacy lab examples to OpenAI Chat Completions."""

from __future__ import annotations

from copy import deepcopy


OPENAI_MODEL_MAP = {
    "claude-haiku-4-5-20251001": "gpt-5.4-nano",
    "claude-sonnet-4-6": "gpt-5.4-mini",
    "claude-opus-4-5": "gpt-5.4",
}


def map_model_name(model_name: str) -> str:
    """Map legacy model names used in the labs to OpenAI model names."""
    return OPENAI_MODEL_MAP.get(model_name, model_name)


def convert_anthropic_tools_to_openai(tools: list[dict] | None) -> list[dict] | None:
    """Convert legacy tool definitions to OpenAI Chat Completions tools."""
    if not tools:
        return tools

    converted = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            converted.append(deepcopy(tool))
            continue

        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": deepcopy(tool.get("input_schema", {"type": "object", "properties": {}})),
                },
            }
        )
    return converted


def build_openai_messages(messages: list[dict], system: str = "") -> list[dict]:
    """Build OpenAI-compatible messages, injecting a system message when present."""
    if system:
        return [{"role": "system", "content": system}, *messages]
    return list(messages)


def create_chat_completion(
    client,
    model: str,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 1000,
    tools: list[dict] | None = None,
    temperature: float | None = None,
):
    """Call OpenAI Chat Completions with lab-friendly defaults."""
    resolved_model = map_model_name(model)
    kwargs = {
        "model": resolved_model,
        "messages": build_openai_messages(messages, system=system),
    }
    if resolved_model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = convert_anthropic_tools_to_openai(tools)
        kwargs["tool_choice"] = "auto"
    if temperature is not None:
        kwargs["temperature"] = temperature
    return client.chat.completions.create(**kwargs)


def completion_text(completion) -> str:
    """Extract plain text from a Chat Completions response."""
    return completion.choices[0].message.content or ""


def finish_reason(completion) -> str | None:
    return completion.choices[0].finish_reason


def prompt_tokens(completion) -> int:
    return getattr(completion.usage, "prompt_tokens", 0) or 0


def completion_tokens(completion) -> int:
    return getattr(completion.usage, "completion_tokens", 0) or 0


def assistant_message_with_tool_calls(message) -> dict:
    """Recreate the assistant tool-call message for the next API round."""
    tool_calls = []
    for tool_call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": tool_calls,
    }
