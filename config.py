from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

YAML_PATH = Path(__file__).parent / "system.yaml"
ENV_PATH = Path(__file__).parent / ".env"


class EnvSettings(BaseSettings):
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek API Key")
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略 .env 中未定義的變數
    )


class AppConfig(BaseSettings):
    # ollama
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Ollama 服務位址",
    )
    ollama_embedding_model: str | None = Field(
        default=None,
        description="Ollama Embedding 模型名稱",
    )
    ollama_num_ctx: int = Field(
        default=8192,
        description="Ollama context window 大小（num_ctx），透傳給 /v1 endpoint",
    )

    # deepseek
    deepseek_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API 位址",
    )
    deepseek_llm_model: str = Field(
        default="deepseek-v4-flash",
        description="DeepSeek 主推理模型名稱",
    )
    deepseek_ingest_model: str = Field(
        default="deepseek-v4-flash",
        description="DeepSeek 資料處理 ingest 模型名稱",
    )

    prompt: str | None = Field(
        default="",
    )

    model_config = SettingsConfigDict(
        yaml_file=YAML_PATH,
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


env_settings = EnvSettings()
app_settings = AppConfig()
