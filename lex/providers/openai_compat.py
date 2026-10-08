"""Any OpenAI-compatible /chat/completions endpoint (OpenAI, OpenRouter, Ollama, LM Studio, vLLM...)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from lex.providers.base import ProviderError, ToolCall, Turn, with_retries


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str, temperature: float | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self._url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(300, connect=15))

    def start_chat(self, role: str, system: str, tools: list[dict[str, Any]]) -> "OpenAIChat":
        return OpenAIChat(self, system, tools)


class OpenAIChat:
    def __init__(self, provider: OpenAICompatProvider, system: str, tools: list[dict[str, Any]]) -> None:
        self._p = provider
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._tools = [{"type": "function", "function": t} for t in tools]

    async def send(self, text: str) -> Turn:
        self._messages.append({"role": "user", "content": text})
        return await self._generate()

    async def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> Turn:
        for call, result in results:
            self._messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        return await self._generate()

    async def _generate(self) -> Turn:
        body: dict[str, Any] = {"model": self._p.model, "messages": self._messages}
        if self._tools:
            body["tools"] = self._tools
        if self._p.temperature is not None:
            body["temperature"] = self._p.temperature

        async def call():
            try:
                r = await self._p._http.post(self._p._url, json=body)
            except httpx.HTTPError as e:
                raise ProviderError(f"Request failed: {e}", retryable=True) from e
            if r.status_code >= 400:
                raise ProviderError(f"API error {r.status_code}: {r.text[:500]}",
                                    retryable=r.status_code in (408, 409, 429) or r.status_code >= 500)
            return r.json()

        data = await with_retries(call)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"Unexpected response: {str(data)[:300]}") from e
        self._messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": fn.get("arguments")}
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), args=args if isinstance(args, dict) else {}))
        usage = data.get("usage") or {}
        return Turn(text=(msg.get("content") or "").strip(), tool_calls=calls,
                    input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0))
