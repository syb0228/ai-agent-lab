"""
common/openai_compat.py
==================================================================
OpenAI Chat Completions API 호출을 감싸는 얇은 호환 헬퍼.

목적:
  - system 프롬프트를 messages 앞에 자동으로 합쳐준다.
  - 응답 객체에서 자주 쓰는 값(finish_reason, 본문 텍스트)을 꺼내는
    접근을 한 곳으로 모은다.
  - tool_calls 가 포함된 assistant 메시지를 다음 턴 messages 에 다시
    넣을 수 있는 dict 형태로 변환한다.
"""

from typing import Any, Optional


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


def create_chat_completion(
    client: Any,
    model: str,
    messages: list[dict],
    system: Optional[str] = None,
    tools: Optional[list] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    """Chat Completions 호출. system 이 있으면 messages 앞에 붙인다."""
    final_messages: list[dict] = []
    if system:
        final_messages.append({"role": "system", "content": system})
    final_messages.extend(messages)

    params: dict[str, Any] = {
        "model": model,
        "messages": final_messages,
    }
    if tools:
        params["tools"] = [_to_openai_tool(t) for t in tools]
        params["tool_choice"] = "auto"
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature

    return client.chat.completions.create(**params)


def finish_reason(response: Any) -> str:
    """응답의 종료 사유(stop, tool_calls 등)를 반환한다."""
    return response.choices[0].finish_reason


def completion_text(response: Any) -> str:
    """응답의 본문 텍스트를 반환한다(없으면 빈 문자열)."""
    return response.choices[0].message.content or ""


def assistant_message_with_tool_calls(message: Any) -> dict:
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
