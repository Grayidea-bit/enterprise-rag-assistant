from functools import cache

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import app_settings, env_settings


def _model(*, model_name: str) -> OpenAIChatModel:
    if not env_settings.LLM_API_KEY:
        raise RuntimeError("Missing the api key of LLM (NVIDIA NIM)")
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=env_settings.LLM_URL, api_key=env_settings.LLM_API_KEY
        ),
    )


@cache
def get_model() -> OpenAIChatModel:
    """延遲建構 NIM chat model;第一次取用時才檢查金鑰並建立(之後快取)"""
    return _model(model_name=app_settings.llm_model_name)
