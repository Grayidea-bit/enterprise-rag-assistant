"""檢索與問答端點的整合行為。

排序品質不在這裡驗 —— TestEmbeddingModel 對任何輸入都回傳同一個向量,
所有段落的距離都是 0。這裡確認的是租戶界線、參數、事件順序與旁路資料。
"""

import json

import pytest

from tests.integration.conftest import upload

pytestmark = pytest.mark.integration

DOC = "國內出差住宿費核實報支,每日上限為新臺幣二千八百元。膳雜費每日上限六百元。"


async def seed(client, tenant):
    response = await upload(client, tenant, "travel.md", DOC)
    assert response.status_code == 201, response.text


class TestSearch:
    async def test_returns_rows_for_own_tenant(self, client, alice):
        await seed(client, alice)
        response = await client.post(
            "/search", headers=alice.headers, json={"query": "住宿費", "limit": 3}
        )
        assert response.status_code == 200
        hits = response.json()
        assert hits and all(h["source"] == "travel.md" for h in hits)
        assert all({"source", "title", "distance", "excerpt"} <= h.keys() for h in hits)

    async def test_other_tenant_sees_nothing(self, client, alice, bob):
        await seed(client, alice)
        response = await client.post("/search", headers=bob.headers, json={"query": "住宿費"})
        assert response.json() == []

    async def test_limit_is_honoured(self, client, alice):
        long_doc = "".join(f"第{i}條規定。" for i in range(1, 80))
        await upload(client, alice, "rules.md", long_doc, chunk_size=100, overlap=0)
        hits = (
            await client.post("/search", headers=alice.headers, json={"query": "規定", "limit": 2})
        ).json()
        assert len(hits) <= 2

    @pytest.mark.parametrize("mode", ["vector", "hybrid"])
    async def test_both_retrieval_modes_work(self, client, alice, mode):
        await seed(client, alice)
        response = await client.post(
            "/search", headers=alice.headers, json={"query": "住宿費", "mode": mode}
        )
        assert response.status_code == 200
        assert response.json()

    async def test_invalid_mode_is_rejected(self, client, alice):
        response = await client.post(
            "/search", headers=alice.headers, json={"query": "x", "mode": "bogus"}
        )
        assert response.status_code == 422


class TestChat:
    async def test_returns_answer_sources_and_conversation(self, client, alice):
        await seed(client, alice)
        response = await client.post(
            "/chat", headers=alice.headers, json={"question": "住宿費上限?"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answer"]
        assert isinstance(body["conversation_id"], int)
        # TestModel 一定會呼叫工具,所以 sources 有值就證明 RagDeps 那條旁路是通的
        assert body["sources"], "工具跑了卻沒把來源傳回端點"
        assert body["sources"][0]["source"] == "travel.md"

    async def test_conversation_is_reused_when_supplied(self, client, alice):
        await seed(client, alice)
        first = (
            await client.post("/chat", headers=alice.headers, json={"question": "第一題"})
        ).json()
        second = (
            await client.post(
                "/chat",
                headers=alice.headers,
                json={"question": "第二題", "conversation_id": first["conversation_id"]},
            )
        ).json()
        assert second["conversation_id"] == first["conversation_id"]

    async def test_cannot_continue_another_tenants_conversation(self, client, alice, bob):
        await seed(client, alice)
        mine = (await client.post("/chat", headers=alice.headers, json={"question": "問題"})).json()
        response = await client.post(
            "/chat",
            headers=bob.headers,
            json={"question": "偷看", "conversation_id": mine["conversation_id"]},
        )
        assert response.status_code == 404

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": ""},
            {"question": "x", "limit": 0},
            {"question": "x", "limit": 999},
            {"question": "x", "max_distance": -1},
        ],
    )
    async def test_invalid_payloads(self, client, alice, payload):
        response = await client.post("/chat", headers=alice.headers, json=payload)
        assert response.status_code == 422


class TestStreaming:
    async def test_event_order(self, client, alice):
        await seed(client, alice)
        events: list[tuple[str, dict]] = []
        async with client.stream(
            "POST", "/chat/stream", headers=alice.headers, json={"question": "住宿費?"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            name = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: ") and name:
                    events.append((name, json.loads(line[6:])))

        kinds = [n for n, _ in events]
        assert "error" not in kinds
        assert kinds[:2] == ["conversation", "sources"]
        assert kinds[-1] == "done"
        assert "delta" in kinds

    async def test_bad_conversation_id_returns_404_not_a_stream(self, client, alice):
        """對話解析刻意放在 SSE generator 之外,壞的 id 才回得了真正的狀態碼。"""
        response = await client.post(
            "/chat/stream",
            headers=alice.headers,
            json={"question": "x", "conversation_id": 99999999},
        )
        assert response.status_code == 404
