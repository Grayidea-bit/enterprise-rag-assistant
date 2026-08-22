"""認證的整合行為。"""

import pytest

from core.auth import generate_key
from database.func import insert_api_key, revoke_api_key
from tests.integration.conftest import Tenant

pytestmark = pytest.mark.integration


async def test_missing_key_is_rejected(client):
    response = await client.get("/documents")
    assert response.status_code == 401
    assert "bearer" in response.headers.get("www-authenticate", "").lower()


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer not-a-real-key"},
        {"Authorization": "Bearer "},
        {"Authorization": "Basic abc"},
        {"X-API-Key": "erag_definitely-not-registered"},
    ],
)
async def test_invalid_keys_are_rejected(client, headers):
    assert (await client.get("/documents", headers=headers)).status_code == 401


async def test_key_determines_tenant(client, alice):
    response = await client.get("/me", headers=alice.headers)
    assert response.status_code == 200
    assert response.json() == {"tenant_id": alice.id, "auth_mode": "api_key"}


async def test_tenant_header_cannot_override_the_key(client, alice, bob):
    """帶著合法金鑰再宣稱別的租戶 —— 這是這整套隔離最該擋住的攻擊。"""
    await upload_doc(client, alice, "只有 alice 看得到的內容。")
    spoofed = {**bob.headers, "X-Tenant-Id": alice.id}
    response = await client.get("/documents", headers=spoofed)
    assert response.status_code == 200
    assert response.json() == []


async def test_revoked_key_stops_working_immediately(client, alice):
    key, key_hash, prefix = generate_key()
    key_id = await insert_api_key(key_hash, alice.id, "temp", prefix)
    headers = {"Authorization": f"Bearer {key}"}

    assert (await client.get("/documents", headers=headers)).status_code == 200
    assert await revoke_api_key(key_id) is True
    assert (await client.get("/documents", headers=headers)).status_code == 401
    assert await revoke_api_key(key_id) is False


async def test_x_api_key_header_works(client, alice):
    key, key_hash, prefix = generate_key()
    await insert_api_key(key_hash, alice.id, "temp", prefix)
    response = await client.get("/documents", headers={"X-API-Key": key})
    assert response.status_code == 200


async def upload_doc(client, tenant: Tenant, body: str, name: str = "doc.md"):
    response = await client.post(
        "/documents",
        headers=tenant.headers,
        files={"file": (name, body.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 201, response.text
    return response.json()
