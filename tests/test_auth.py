"""core/auth.py 的單元測試。"""

from core.auth import KEY_PREFIX, extract_key, generate_key, hash_key


def test_generated_keys_are_unique_and_prefixed():
    keys = {generate_key()[0] for _ in range(200)}
    assert len(keys) == 200
    assert all(k.startswith(KEY_PREFIX) for k in keys)


def test_generated_key_has_enough_entropy():
    key, _, _ = generate_key()
    # token_urlsafe(32) 會產生 43 個字元
    assert len(key) - len(KEY_PREFIX) >= 40


def test_hash_is_stable_and_irreversible_looking():
    key, key_hash, _ = generate_key()
    assert key_hash == hash_key(key)
    assert len(key_hash) == 64 and key not in key_hash


def test_different_keys_hash_differently():
    a, _, _ = generate_key()
    b, _, _ = generate_key()
    assert hash_key(a) != hash_key(b)


def test_prefix_is_short_and_not_the_whole_key():
    key, _, prefix = generate_key()
    assert key.startswith(prefix) and len(prefix) < len(key) / 2


class TestExtractKey:
    def test_bearer_header(self):
        assert extract_key("Bearer abc123", None) == "abc123"

    def test_bearer_scheme_is_case_insensitive(self):
        assert extract_key("bearer abc123", None) == "abc123"
        assert extract_key("BEARER abc123", None) == "abc123"

    def test_surrounding_whitespace_is_trimmed(self):
        assert extract_key("Bearer   abc123  ", None) == "abc123"

    def test_x_api_key_fallback(self):
        assert extract_key(None, "abc123") == "abc123"

    def test_authorization_wins_over_x_api_key(self):
        assert extract_key("Bearer from-auth", "from-header") == "from-auth"

    def test_non_bearer_scheme_falls_through(self):
        assert extract_key("Basic abc123", None) is None
        assert extract_key("Basic abc123", "fallback") == "fallback"

    def test_empty_and_missing_values(self):
        assert extract_key(None, None) is None
        assert extract_key("", "") is None
        assert extract_key("Bearer ", None) is None
        assert extract_key("Bearer    ", "  ") is None
