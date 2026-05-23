from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import app_settings


def _ollama_provider() -> OpenAIProvider:
    base = (app_settings.ollama_url or "http://localhost:11434").rstrip("/")
    return OpenAIProvider(base_url=f"{base}/v1", api_key="ollama")


def _ollama_model(*, model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(model_name, provider=_ollama_provider())


def _anthropic_model(*, model_name: str) -> AnthropicModel:
    if not env_settings.CLAUDE_API_KEY:
        raise RuntimeError("llm_provider=anthropic 但 .env 未設定 CLAUDE_API_KEY")
    return AnthropicModel(
        model_name,
        provider=AnthropicProvider(api_key=env_settings.CLAUDE_API_KEY),
    )


def _build_chat_pair() -> tuple[Model, Model]:
    """回傳 (thinking_model, fast_model)。anthropic/openai 共用同一物件。"""
    provider = app_settings.llm_provider
    if provider == "anthropic":
        shared = _anthropic_model(model_name=app_settings.d)
        return shared, shared
    if provider == "openai":
        shared = _openai_model(model_name=app_settings.openai_model_name)
        return shared, shared
    # ollama：thinking / fast 兩個實例（reasoning 旗標由呼叫端在 model_settings 傳）
    return (
        _ollama_model(model_name=app_settings.model_name),
        _ollama_model(model_name=app_settings.model_name),
    )


model_thinking, model_fast = _build_chat_pair()
