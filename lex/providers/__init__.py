from lex.providers.base import Chat, Provider, ProviderError, ToolCall, Turn

__all__ = ["Chat", "Provider", "ProviderError", "ToolCall", "Turn", "create_provider"]


def create_provider(settings) -> Provider:
    """Build the provider named in settings."""
    if settings.provider == "gemini":
        from lex.providers.gemini import GeminiProvider

        return GeminiProvider(settings.model, settings.api_key(), settings.temperature)
    if settings.provider == "openai":
        from lex.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(settings.model, settings.api_key(), settings.base_url, settings.temperature)
    raise ProviderError(f"Unknown provider '{settings.provider}'. Use 'gemini' or 'openai'.")
