"""LLM model factory for pydantic-ai.

供 `core/agent.py` 與 `services/*` 取得 pydantic-ai `Model` 實例。
Provider 切換邏輯支援四家：ollama / deepseek / anthropic / openai。
Ollama 與 DeepSeek 都走 OpenAI-compatible endpoint（DeepSeek 原生相容，
Ollama 透過 /v1）。
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from config import app_settings, env_settings

# 走 OpenAI-compatible 協議的 provider（沒有原生 pydantic-ai provider）
_OPENAI_COMPATIBLE = ("ollama", "deepseek")


# ---------------------------------------------------------------------------
# provider 對應的 model 建構子
# ---------------------------------------------------------------------------
def _ollama_provider() -> OpenAIProvider:
    base = (app_settings.ollama_base_url or "http://localhost:11434").rstrip("/")
    return OpenAIProvider(base_url=f"{base}/v1", api_key="ollama")


def _ollama_model(*, model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(model_name, provider=_ollama_provider())


def _deepseek_provider() -> OpenAIProvider:
    if not env_settings.DEEPSEEK_API_KEY:
        raise RuntimeError("provider=deepseek 但 .env 未設定 DEEPSEEK_API_KEY")
    base = (app_settings.deepseek_url or "https://api.deepseek.com").rstrip("/")
    return OpenAIProvider(base_url=base, api_key=env_settings.DEEPSEEK_API_KEY)


def _deepseek_model(*, model_name: str) -> OpenAIChatModel:
    return OpenAIChatModel(model_name, provider=_deepseek_provider())


def _anthropic_model(*, model_name: str) -> AnthropicModel:
    if not env_settings.ANTHROPIC_API_KEY:
        raise RuntimeError("provider=anthropic 但 .env 未設定 ANTHROPIC_API_KEY")
    return AnthropicModel(
        model_name,
        provider=AnthropicProvider(api_key=env_settings.ANTHROPIC_API_KEY),
    )


def _openai_model(*, model_name: str) -> OpenAIChatModel:
    if not env_settings.OPENAI_API_KEY:
        raise RuntimeError("provider=openai 但 .env 未設定 OPENAI_API_KEY")
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(api_key=env_settings.OPENAI_API_KEY),
    )


# ---------------------------------------------------------------------------
# provider -> (chat model 名稱, ingest model 名稱) 解析
# ---------------------------------------------------------------------------
def _chat_model_name(provider: str) -> str:
    name = {
        "ollama": app_settings.ollama_llm_model,
        "deepseek": app_settings.deepseek_llm_model,
        "openai": app_settings.openai_llm_model,
        "anthropic": app_settings.anthropic_llm_model,
    }.get(provider)
    if not name:
        raise RuntimeError(f"llm_provider={provider} 但對應的 *_llm_model 未設定")
    return name


def _ingest_model_name(provider: str) -> str:
    name = {
        "ollama": app_settings.ollama_ingest_model,
        "deepseek": app_settings.deepseek_ingest_model,
        "openai": app_settings.openai_ingest_model,
        "anthropic": app_settings.anthropic_ingest_model,
    }.get(provider)
    if not name:
        raise RuntimeError(f"ingest_provider={provider} 但對應的 *_ingest_model 未設定")
    return name


def _make_model(provider: str, *, model_name: str) -> Model:
    if provider == "anthropic":
        return _anthropic_model(model_name=model_name)
    if provider == "openai":
        return _openai_model(model_name=model_name)
    if provider == "deepseek":
        return _deepseek_model(model_name=model_name)
    return _ollama_model(model_name=model_name)


# ---------------------------------------------------------------------------
# 對外建構：chat pair 與 ingest model
# ---------------------------------------------------------------------------
def _build_chat_pair() -> tuple[Model, Model]:
    """回傳 (thinking_model, fast_model)。

    非 ollama provider 目前 thinking / fast 共用同一物件（reasoning 由
    呼叫端在 model_settings 控制）；ollama 則建兩個獨立實例。
    """
    provider = app_settings.llm_provider
    model_name = _chat_model_name(provider)
    if provider == "ollama":
        return (
            _ollama_model(model_name=model_name),
            _ollama_model(model_name=model_name),
        )
    shared = _make_model(provider, model_name=model_name)
    return shared, shared


def _build_ingest_model() -> Model:
    provider = app_settings.ingest_provider
    return _make_model(provider, model_name=_ingest_model_name(provider))


model_thinking, model_fast = _build_chat_pair()
ingest_model = _build_ingest_model()


# ---------------------------------------------------------------------------
# 輔助
# ---------------------------------------------------------------------------
def llm_chat_settings(*, reasoning: bool) -> ModelSettings:
    """供 agent.run/iter 傳給 OpenAI-compatible provider 的 model_settings。

    `num_ctx` 與 `reasoning` 都是 Ollama 特有；OpenAIChatModel 用 extra_body
    透傳給 /v1/chat/completions，Ollama 端會讀。anthropic/openai provider
    時呼叫端不需要這個。

    Ollama 在較新版本將 `reasoning` 改為與 OpenAI 一致的 object 形式
    （`{"effort": "low|medium|high"}`），不再接受布林值，否則會回
    `json: cannot unmarshal bool into Go struct field ... of type openai.Reasoning`。
    """
    extra_body: dict = {"options": {"num_ctx": app_settings.ollama_num_ctx}}
    if reasoning:
        extra_body["reasoning"] = {"effort": "medium"}
    return ModelSettings(extra_body=extra_body)


def is_ollama() -> bool:
    return app_settings.llm_provider == "ollama"


def is_openai_compatible() -> bool:
    """provider 是否走 OpenAI-compatible body（ollama / deepseek）。"""
    return app_settings.llm_provider in _OPENAI_COMPATIBLE


def current_chat_model_name() -> str:
    """回傳目前 chat agent 實際使用的 model 名稱（依 provider）。

    供 logging / SSE 記錄 `model_used`，避免呼叫端各自寫 provider 判斷。
    """
    return _chat_model_name(app_settings.llm_provider)
