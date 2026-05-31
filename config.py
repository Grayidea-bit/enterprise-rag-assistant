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
    LLM_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="LLM CONN URL（NVIDIA NIM OpenAI 相容端點）",
    )
    LLM_API_KEY: str = Field(default="", description="LLM API Key（NIM nvapi- 金鑰）")
    DATABASE_URL: str = Field(
        default="postgresql://graytsao@localhost:5432/enterprise_rag",
    )

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略 .env 中未定義的變數
    )


class AppConfig(BaseSettings):
    # NVIDIA NIM（chat 與 embedding 共用同一個 OpenAI 相容端點與金鑰）
    llm_model_name: str | None = Field(
        default=None,
        description="NIM Chat 模型名稱",
    )
    embedding_model_name: str | None = Field(
        default=None,
        description="NIM Embedding 模型名稱",
    )
    embedding_dim: int = Field(
        default=1024,
        description="Embedding 向量維度（需與 schema 的 VECTOR(n) 一致）",
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
