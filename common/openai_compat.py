"""
common/openai_compat.py
==================================================================
OpenAI Chat Completions API 호출을 감싸는 얇은 호환 헬퍼.

목적:
  - system 프롬프트를 messages 앞에 자동으로 합쳐준다.
  - 응답 객체에서 자주 쓰는 값(finish_reason, 본문 텍스트, 토큰 수)을
    꺼내는 접근을 한 곳으로 모은다.
  - Anthropic 형식 tools(input_schema)를 OpenAI 형식(parameters)으로 변환한다.
  - tool_calls 가 포함된 assistant 메시지를 다음 턴 messages 에 다시
    넣을 수 있는 dict 형태로 변환한다.
"""


def _to_openai_tool(tool: dict) -> dict:
    """Anthropic 형식(name/description/input_schema) tool 을 OpenAI 형식으로 변환한다.
    이미 OpenAI 형식({"type": "function", ...})이면 그대로 반환한다."""
    if tool.get("type") == "function" and "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", tool.get("parameters", {})),
        },
    }


def create_chat_completion(client, model, max_tokens, system, messages, temperature=0.2, tools=None):
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": temperature,
    }
    if tools:
        # Anthropic 형식(input_schema)이 섞여 있어도 OpenAI 형식으로 변환한다.
        kwargs["tools"] = [_to_openai_tool(t) for t in tools]
    return client.chat.completions.create(**kwargs)


def finish_reason(response):
    return response.choices[0].finish_reason


def completion_text(response):
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def assistant_message_with_tool_calls(message) -> dict:
    """tool_calls 가 포함된 assistant 메시지를 messages 용 dict 로 변환한다."""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in (message.tool_calls or [])
        ],
    }


def prompt_tokens(response):
    usage = getattr(response, "usage", None)
    return getattr(usage, "prompt_tokens", 0) if usage else 0


def completion_tokens(response):
    usage = getattr(response, "usage", None)
    return getattr(usage, "completion_tokens", 0) if usage else 0
