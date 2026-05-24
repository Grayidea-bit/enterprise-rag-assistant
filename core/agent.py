from pydantic_ai import Agent

from config import app_settings
from core.llm import model

tools = []


def _build_agent(model) -> Agent:
    return Agent(model, instructions=app_settings.prompt, tools=tools)


agent = _build_agent(model)
