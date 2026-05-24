from llm import model
from pydantic_ai import Agent

from config import app_settings

tools = []


def _build_agent(model) -> Agent:
    return Agent(model, instructions=app_settings.prompt, tools=tools)


agent = _build_agent(model)
