"""API 金鑰的產生與驗證。

金鑰只在產生的當下回傳一次明文,資料庫裡只留 SHA-256 雜湊。
"""

import hashlib
import secrets

KEY_PREFIX = "erag_"
KEY_BYTES = 32  # 256 bits 的熵


def generate_key() -> tuple[str, str, str]:
    """回傳 (明文金鑰, 雜湊, 顯示用前綴)。明文只有這一次機會拿到。"""
    token = secrets.token_urlsafe(KEY_BYTES)
    key = f"{KEY_PREFIX}{token}"
    return key, hash_key(key), key[: len(KEY_PREFIX) + 6]


def hash_key(key: str) -> str:
    """金鑰是高熵隨機值,用快雜湊即可;慢雜湊(bcrypt/argon2)是給低熵密碼用的。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """從 Authorization: Bearer <key> 或 X-API-Key header 取出金鑰。"""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return None
