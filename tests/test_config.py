"""config.py 的設定解析與 fallback 規則。"""

import pytest

from config import EnvSettings

BASE = {
    "CHAT_BASE_URL": "https://chat.example/v1",
    "CHAT_MODEL": "chat-model",
    "EMBEDDING_MODEL": "embed-model",
}


def settings(**overrides) -> EnvSettings:
    # _env_file=None 讓測試不受本機 .env 影響
    return EnvSettings(_env_file=None, **{**BASE, **overrides})


class TestRequiredFields:
    @pytest.mark.parametrize("missing", ["CHAT_BASE_URL", "CHAT_MODEL", "EMBEDDING_MODEL"])
    def test_missing_required_field_raises(self, missing):
        kwargs = {k: v for k, v in BASE.items() if k != missing}
        with pytest.raises(Exception, match=missing):
            EnvSettings(_env_file=None, **kwargs)


class TestChatTarget:
    def test_empty_api_key_becomes_none(self):
        """OpenAIProvider 只在 api_key is None 時才補預設值,空字串會送出空金鑰。"""
        assert settings(CHAT_API_KEY="").chat_target == ("https://chat.example/v1", None)

    def test_api_key_is_passed_through(self):
        assert settings(CHAT_API_KEY="sk-1").chat_target[1] == "sk-1"


class TestEmbeddingTarget:
    def test_falls_back_to_chat_endpoint_and_key(self):
        s = settings(CHAT_API_KEY="sk-chat")
        assert s.embedding_target == ("https://chat.example/v1", "sk-chat")
        assert s.embedding_endpoint_is_shared is True

    def test_explicit_endpoint_does_not_inherit_the_chat_key(self):
        """跨主機不該沿用金鑰 —— 那會把 chat 的金鑰送到別人家。"""
        s = settings(CHAT_API_KEY="sk-chat", EMBEDDING_BASE_URL="https://embed.example/v1")
        assert s.embedding_target == ("https://embed.example/v1", None)
        assert s.embedding_endpoint_is_shared is False

    def test_explicit_endpoint_uses_its_own_key(self):
        s = settings(
            CHAT_API_KEY="sk-chat",
            EMBEDDING_BASE_URL="https://embed.example/v1",
            EMBEDDING_API_KEY="sk-embed",
        )
        assert s.embedding_target == ("https://embed.example/v1", "sk-embed")

    def test_own_key_with_shared_endpoint_wins_over_chat_key(self):
        s = settings(CHAT_API_KEY="sk-chat", EMBEDDING_API_KEY="sk-embed")
        assert s.embedding_target == ("https://chat.example/v1", "sk-embed")


class TestDefaults:
    def test_secure_and_sensible_defaults(self):
        s = settings()
        assert s.AUTH_MODE == "api_key", "預設就該要求認證"
        assert s.AUTO_MIGRATE is False, "預設不該在啟動時自動改 schema"
        assert s.RETRIEVAL_MODE == "hybrid"
        assert s.EMBEDDING_DIM == 1024
        assert s.LLM_TIMEOUT_SECONDS < 600, "必須比 OpenAI SDK 的預設短"

    @pytest.mark.parametrize("bad", ["always", "none", ""])
    def test_invalid_auth_mode_rejected(self, bad):
        with pytest.raises(Exception):
            settings(AUTH_MODE=bad)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_non_positive_timeout_rejected(self, bad):
        with pytest.raises(Exception):
            settings(LLM_TIMEOUT_SECONDS=bad)
