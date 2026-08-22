from pathlib import Path
from typing import Literal

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
    """所有連線相關設定（端點、金鑰、模型名稱）一律由環境變數 / .env 驅動。

    這樣「換一個 OpenAI 相容的 provider」就只是改 .env 的一件事，
    不必動到程式碼或 system.yaml。
    """

    # ── Chat（必填）────────────────────────────────────────────────
    CHAT_BASE_URL: str = Field(
        description="OpenAI 相容 chat 端點，多數 provider 需以 /v1 結尾",
    )
    CHAT_API_KEY: str = Field(
        default="",
        description="Chat 金鑰；自架服務（Ollama / vLLM / LM Studio）可留空",
    )
    CHAT_MODEL: str = Field(description="Chat 模型名稱")

    # ── Embedding（端點與金鑰選填，留空則沿用 chat）──────────────────
    EMBEDDING_BASE_URL: str = Field(
        default="",
        description="Embedding 端點；留空則沿用 CHAT_BASE_URL",
    )
    EMBEDDING_API_KEY: str = Field(
        default="",
        description="Embedding 金鑰；fallback 規則見 embedding_target",
    )
    EMBEDDING_MODEL: str = Field(description="Embedding 模型名稱")
    EMBEDDING_DIM: int = Field(
        default=1024,
        description="Embedding 向量維度（需與 schema.sql 的 VECTOR(n) 一致）",
    )

    # OpenAI SDK 預設 read timeout 是 600 秒,對一個 HTTP 端點來說太長了 ——
    # 端點掛住十分鐘,呼叫端只會看到連線一直不回應。
    LLM_TIMEOUT_SECONDS: float = Field(
        default=120.0,
        gt=0,
        description="單次 LLM / embedding 請求的逾時秒數",
    )

    # agent 單次 run 的煞車。pydantic-ai 預設 request_limit=50、
    # tool_calls_limit 無上限,失控的工具迴圈可以燒掉大量 token。
    AGENT_REQUEST_LIMIT: int = Field(default=6, ge=1)
    AGENT_TOOL_CALLS_LIMIT: int = Field(default=4, ge=1)

    # vector = 只用向量;hybrid = 向量 + trigram 詞彙,用 RRF 融合
    RETRIEVAL_MODE: Literal["vector", "hybrid"] = Field(
        default="hybrid",
        description="檢索模式",
    )

    DATABASE_URL: str = Field(
        default="postgresql://graytsao@localhost:5432/enterprise_rag",
    )

    # api_key = 必須帶有效金鑰,租戶由金鑰決定(X-Tenant-Id 一律忽略)
    # disabled = 開發模式,直接採信 X-Tenant-Id。不要用在任何對外環境。
    AUTH_MODE: Literal["api_key", "disabled"] = Field(
        default="api_key",
        description="認證模式",
    )

    # 只在 AUTH_MODE=disabled 時有意義:X-Tenant-Id 未帶時的預設租戶
    DEFAULT_TENANT_ID: str = Field(
        default="default",
        description="開發模式下 X-Tenant-Id 未帶時使用的租戶",
    )

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略 .env 中未定義的變數
    )

    @property
    def chat_target(self) -> tuple[str, str | None]:
        """Chat 的 (base_url, api_key)。空字串轉 None，讓免金鑰端點能正常運作。"""
        return self.CHAT_BASE_URL, (self.CHAT_API_KEY or None)

    @property
    def embedding_target(self) -> tuple[str, str | None]:
        """Embedding 的 (base_url, api_key)。

        金鑰只在「端點也是沿用 chat 的」情況下才跟著沿用；
        一旦明確指定了不同的 EMBEDDING_BASE_URL，就不把 chat 的金鑰送到別台主機。
        """
        if self.EMBEDDING_BASE_URL:
            return self.EMBEDDING_BASE_URL, (self.EMBEDDING_API_KEY or None)
        return self.CHAT_BASE_URL, (self.EMBEDDING_API_KEY or self.CHAT_API_KEY or None)

    @property
    def embedding_endpoint_is_shared(self) -> bool:
        """Embedding 端點是否沿用 chat 的（診斷用）。"""
        return not self.EMBEDDING_BASE_URL


class AppConfig(BaseSettings):
    """純應用層設定，來自 system.yaml。不放任何連線資訊。"""

    prompt: str | None = Field(
        default="",
        description="Agent 的 system prompt",
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


env_settings = EnvSettings()  # type: ignore[call-arg]
app_settings = AppConfig()
