"""Google Gemini via the google-genai SDK."""

from __future__ import annotations

import uuid
from typing import Any

from lex.providers.base import ProviderError, ToolCall, Turn, with_retries


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: str, temperature: float | None = None) -> None:
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise ProviderError("google-genai is not installed: pip install google-genai") from e
        if not api_key:
            raise ProviderError("No Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        self.model = model
        self.temperature = temperature
        self._client = genai.Client(api_key=api_key)

    def start_chat(self, role: str, system: str, tools: list[dict[str, Any]]) -> "GeminiChat":
        return GeminiChat(self, system, tools)


class GeminiChat:
    def __init__(self, provider: GeminiProvider, system: str, tools: list[dict[str, Any]]) -> None:
        from google.genai import types

        self._types = types
        self._p = provider
        self._history: list[Any] = []
        declarations = [
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["parameters"])
            for t in tools
        ]
        self._config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=provider.temperature,
        )

    async def send(self, text: str) -> Turn:
        t = self._types
        self._history.append(t.Content(role="user", parts=[t.Part.from_text(text=text)]))
        return await self._generate()

    async def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> Turn:
        t = self._types
        parts = []
        for call, result in results:
            part = t.Part.from_function_response(name=call.name, response=result)
            if part.function_response is not None and not call.id.startswith("lex-"):
                part.function_response.id = call.id
            parts.append(part)
        self._history.append(t.Content(role="user", parts=parts))
        return await self._generate()

    async def _generate(self) -> Turn:
        from google.genai import errors

        async def call():
            try:
                return await self._p._client.aio.models.generate_content(
                    model=self._p.model, contents=self._history, config=self._config)
            except errors.APIError as e:
                code = getattr(e, "code", 0) or 0
                raise ProviderError(f"Gemini API error {code}: {e.message or e}",
                                    retryable=code in (408, 429) or code >= 500) from e
            except Exception as e:  # network errors etc.
                raise ProviderError(f"Gemini request failed: {e}", retryable=True) from e

        resp = await with_retries(call)
        candidate = resp.candidates[0] if resp.candidates else None
        if candidate is None or candidate.content is None:
            reason = getattr(candidate, "finish_reason", None) or getattr(resp.prompt_feedback, "block_reason", "unknown")
            # Keep history valid so the agent can be nudged and retried.
            self._history.append(self._types.Content(role="model", parts=[self._types.Part.from_text(text="(no response)")]))
            return Turn(text=f"(The model returned no content: {reason})")

        # Keep the native content so thought signatures survive into the next request.
        self._history.append(candidate.content)
        texts, calls = [], []
        for part in candidate.content.parts or []:
            if part.function_call is not None:
                fc = part.function_call
                calls.append(ToolCall(id=fc.id or f"lex-{uuid.uuid4().hex[:8]}", name=fc.name or "", args=dict(fc.args or {})))
            elif part.text and not part.thought:
                texts.append(part.text)
        usage = resp.usage_metadata
        return Turn(
            text="".join(texts).strip(), tool_calls=calls,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        )
