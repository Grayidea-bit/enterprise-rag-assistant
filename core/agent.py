from functools import cache

from pydantic_ai import Agent

from config import app_settings
from core.llm import get_model

tools = []


def _build_agent(model) -> Agent:
    return Agent(model, instructions=app_settings.prompt, tools=tools)


@cache
def get_agent() -> Agent:
    """延遲建構 agent;第一次取用時才透過 get_model() 建立 model(之後快取)"""
    return _build_agent(get_model())
