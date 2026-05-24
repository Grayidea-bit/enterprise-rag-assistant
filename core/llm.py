from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import app_settings, env_settings


def _model(*, model_name: str) -> OpenAIChatModel:
    if not env_settings.DEEPSEEK_API_KEY:
        raise RuntimeError("Missing the api key of DeepSeek")
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=app_settings.deepseek_url, api_key=env_settings.DEEPSEEK_API_KEY
        ),
    )


model = _model(model_name=app_settings.deepseek_llm_model)
