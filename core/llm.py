from dataclasses import replace
from functools import cache

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

from config import env_settings


def build_provider(base_url: str, api_key: str | None) -> OpenAIProvider:
    """建立指向任意 OpenAI 相容端點的 provider（chat 與 embedding 共用）。

    注意 api_key 必須是 None 而非空字串:OpenAIProvider 只在 api_key is None
    時才自動補上 'api-key-not-set',傳空字串會真的送出一把空金鑰。
    """
    return OpenAIProvider(base_url=base_url, api_key=api_key or None)


def compat_profile(model_name: str) -> ModelProfile:
    """泛用 OpenAI 相容端點的 model profile。

    沿用 pydantic-ai 依模型名稱推斷出來的設定,只覆寫兩個對自架服務不成立的預設。
    傳「函式」而非 instance 很重要:OpenAIChatModel 的 profile 參數是整組取代
    provider 的推斷結果,包成 callable 才能保留其餘欄位。
    """
    return replace(
        openai_model_profile(model_name),
        # strict tool definition 是 OpenAI 專屬擴充,Ollama / vLLM 都不支援
        openai_supports_strict_tool_definition=False,
        # Qwen3 等 thinking 模型把內容放在 reasoning 欄位,不指定會拿到空回覆
        openai_chat_thinking_field="reasoning",
    )


@cache
def get_model() -> OpenAIChatModel:
    """延遲建構 chat model;第一次取用時才建立（之後快取）"""
    base_url, api_key = env_settings.chat_target
    return OpenAIChatModel(
        env_settings.CHAT_MODEL,
        provider=build_provider(base_url, api_key),
        profile=compat_profile,
    )
