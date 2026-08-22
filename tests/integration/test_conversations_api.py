"""對話管理與歷史持久化。"""

import pytest

from database.conn import pool
from tests.integration.conftest import upload

pytestmark = pytest.mark.integration


async def seed(client, tenant):
    assert (
        await upload(client, tenant, "hr.md", "特休滿一年十四天,滿三年十七天。")
    ).status_code == 201


async def ask(client, tenant, question, conversation_id=None):
    payload = {"question": question}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post("/chat", headers=tenant.headers, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


async def test_turns_are_persisted_with_roles_in_order(client, alice):
    await seed(client, alice)
    first = await ask(client, alice, "特休幾天?")
    await ask(client, alice, "那滿三年呢?", first["conversation_id"])

    messages = (
        await client.get(f"/conversations/{first['conversation_id']}", headers=alice.headers)
    ).json()
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "特休幾天?"
    assert messages[2]["content"] == "那滿三年呢?"


async def test_sources_are_stored_on_assistant_messages(client, alice):
    await seed(client, alice)
    conversation = await ask(client, alice, "特休幾天?")
    messages = (
        await client.get(f"/conversations/{conversation['conversation_id']}", headers=alice.headers)
    ).json()
    assert messages[0]["sources"] == []
    assert messages[1]["sources"], "assistant 訊息應該存下當時引用的來源"


async def test_first_question_becomes_the_title(client, alice):
    await seed(client, alice)
    await ask(client, alice, "特休幾天?")
    listing = (await client.get("/conversations", headers=alice.headers)).json()
    assert len(listing) == 1
    assert listing[0]["title"] == "特休幾天?"
    assert listing[0]["message_count"] == 2


async def test_explicit_creation_returns_the_new_conversation(client, alice):
    """回傳的必須是剛建立的那筆,不是「這個租戶最新的一筆」。"""
    created = await client.post("/conversations", headers=alice.headers, json={"title": "手動"})
    assert created.status_code == 201
    body = created.json()
    assert body["title"] == "手動"
    assert body["message_count"] == 0

    fetched = (await client.get("/conversations", headers=alice.headers)).json()
    assert [c["id"] for c in fetched] == [body["id"]]


async def test_conversations_are_scoped_to_the_tenant(client, alice, bob):
    await seed(client, alice)
    mine = await ask(client, alice, "特休幾天?")

    assert (await client.get("/conversations", headers=bob.headers)).json() == []
    assert (
        await client.get(f"/conversations/{mine['conversation_id']}", headers=bob.headers)
    ).status_code == 404
    assert (
        await client.delete(f"/conversations/{mine['conversation_id']}", headers=bob.headers)
    ).status_code == 404


async def test_delete_cascades_to_messages(client, alice):
    await seed(client, alice)
    conversation_id = (await ask(client, alice, "特休幾天?"))["conversation_id"]

    assert (
        await client.delete(f"/conversations/{conversation_id}", headers=alice.headers)
    ).status_code == 204
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = %s", (conversation_id,)
            )
        ).fetchone()
    assert row[0] == 0
    assert (
        await client.get(f"/conversations/{conversation_id}", headers=alice.headers)
    ).status_code == 404
