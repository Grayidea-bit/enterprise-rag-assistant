"""對話歷史 smoke test。

驗證多輪對話:追問時模型看得到前文、對話與訊息都受租戶隔離、
串流也會把回合寫進資料庫。

    python scripts/smoke_conversation.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from database import db_shutdown, db_startup  # noqa: E402
from database.conn import pool  # noqa: E402
from server import app  # noqa: E402

LINE = "─" * 62
TENANT_A = "smoke-conv-a"
TENANT_B = "smoke-conv-b"

DOC = """# 年度休假規定

正職員工每年享有十四天特別休假,到職滿三年後增加為十七天。
特休未休完的部分,可遞延至次年三月三十一日前使用,逾期視為放棄。

## 病假

普通病假每年上限三十天,超過三十天的部分改以事假計算。
住院傷病假合計以一年為限,須檢附醫院診斷證明。
"""

results: list[tuple[str, bool, str]] = []


def ok(name, msg, t):
    print(f"  \033[32m✓\033[0m {msg}  \033[2m({t:.1f}s)\033[0m")
    results.append((name, True, msg))


def fail(name, err, t):
    print(f"  \033[31m✗\033[0m {err}  \033[2m({t:.1f}s)\033[0m")
    results.append((name, False, err))


async def check(name, title, fn):
    print(f"\n{title}")
    t = time.perf_counter()
    try:
        ok(name, await fn(), time.perf_counter() - t)
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}", time.perf_counter() - t)


async def cleanup():
    async with pool.connection() as conn:
        for table in ("conversations", "documents"):
            await conn.execute(
                f"DELETE FROM {table} WHERE tenant_id = ANY(%s)",
                ([TENANT_A, TENANT_B],),
            )


async def main() -> int:
    print(LINE)
    print(" 對話歷史 smoke test")
    print(LINE)

    await db_startup()
    await cleanup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=600
    ) as client:
        a = {"X-Tenant-Id": TENANT_A}
        b = {"X-Tenant-Id": TENANT_B}
        state: dict = {}

        async def t_setup():
            r = await client.post(
                "/documents",
                headers=a,
                files={"file": ("休假規定.md", DOC.encode("utf-8"), "text/markdown")},
                data={"chunk_size": 200, "overlap": 40},
            )
            assert r.status_code == 201, r.text
            return f"灌入 {r.json()['chunks']} 個 chunk"

        async def t_first_turn():
            r = await client.post(
                "/chat", headers=a, json={"question": "特休有幾天?"}
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            body = r.json()
            if not body.get("conversation_id"):
                raise RuntimeError("沒有回傳 conversation_id")
            state["conv"] = body["conversation_id"]
            if "十四" not in body["answer"] and "14" not in body["answer"]:
                raise RuntimeError(f"沒答到天數:{body['answer'][:120]}")
            return f"conversation_id={state['conv']},答案命中天數"

        async def t_follow_up():
            """關鍵測試:追問句本身沒有主詞,只有看得到前文才答得出來。"""
            r = await client.post(
                "/chat",
                headers=a,
                json={"question": "那滿三年之後呢?", "conversation_id": state["conv"]},
            )
            body = r.json()
            answer = body["answer"]
            if body["conversation_id"] != state["conv"]:
                raise RuntimeError("conversation_id 變了")
            if "十七" not in answer and "17" not in answer:
                raise RuntimeError(f"追問沒有接上前文:{answer[:150]}")
            return f"「那滿三年之後呢?」正確答出十七天:{answer[:40]}…"

        async def t_persisted():
            msgs = (await client.get(f"/conversations/{state['conv']}", headers=a)).json()
            if len(msgs) != 4:
                raise RuntimeError(f"應該有 4 則訊息(2 問 2 答),實際 {len(msgs)}")
            roles = [m["role"] for m in msgs]
            if roles != ["user", "assistant", "user", "assistant"]:
                raise RuntimeError(f"角色順序不對: {roles}")
            if not msgs[1]["sources"]:
                raise RuntimeError("assistant 訊息沒有存下來源")
            convs = (await client.get("/conversations", headers=a)).json()
            if len(convs) != 1 or convs[0]["message_count"] != 4:
                raise RuntimeError(f"對話列表不正確: {convs}")
            if not convs[0]["title"]:
                raise RuntimeError("第一輪問題應該被設成標題")
            return f"4 則訊息、來源有存、標題「{convs[0]['title']}」"

        async def t_isolation():
            convs_b = (await client.get("/conversations", headers=b)).json()
            if convs_b:
                raise RuntimeError(f"租戶 B 看到了 A 的對話: {convs_b}")
            r = await client.get(f"/conversations/{state['conv']}", headers=b)
            if r.status_code != 404:
                raise RuntimeError(f"跨租戶讀取應回 404,實際 {r.status_code}")
            r = await client.post(
                "/chat",
                headers=b,
                json={"question": "hi", "conversation_id": state["conv"]},
            )
            if r.status_code != 404:
                raise RuntimeError(f"跨租戶續談應回 404,實際 {r.status_code}")
            return "列表為空、直接讀取 404、續談 404"

        async def t_stream_persists():
            events = []
            async with client.stream(
                "POST", "/chat/stream", headers=a,
                json={"question": "病假上限幾天?", "conversation_id": state["conv"]},
            ) as r:
                name = None
                async for line in r.aiter_lines():
                    if line.startswith("event: "):
                        name = line[7:]
                    elif line.startswith("data: ") and name:
                        events.append((name, json.loads(line[6:])))
            kinds = [n for n, _ in events]
            if kinds[0] != "conversation":
                raise RuntimeError(f"第一個事件應是 conversation,實際 {kinds[:3]}")
            if dict(events)["conversation"]["conversation_id"] != state["conv"]:
                raise RuntimeError("串流回的 conversation_id 不對")
            if "error" in kinds:
                raise RuntimeError(f"串流錯誤: {dict(events)['error']}")
            msgs = (await client.get(f"/conversations/{state['conv']}", headers=a)).json()
            if len(msgs) != 6:
                raise RuntimeError(f"串流後應有 6 則訊息,實際 {len(msgs)}")
            if not msgs[-1]["content"].strip():
                raise RuntimeError("串流寫入的 assistant 訊息是空的")
            return f"事件順序正確,串流回合已寫入(共 {len(msgs)} 則)"

        async def t_delete():
            r = await client.delete(f"/conversations/{state['conv']}", headers=b)
            if r.status_code != 404:
                raise RuntimeError(f"跨租戶刪除應回 404,實際 {r.status_code}")
            r = await client.delete(f"/conversations/{state['conv']}", headers=a)
            if r.status_code != 204:
                raise RuntimeError(f"刪除應回 204,實際 {r.status_code}")
            async with pool.connection() as conn:
                row = await (
                    await conn.execute(
                        "SELECT COUNT(*) FROM messages WHERE conversation_id = %s",
                        (state["conv"],),
                    )
                ).fetchone()
            if row[0] != 0:
                raise RuntimeError(f"CASCADE 沒生效,還剩 {row[0]} 則訊息")
            return "跨租戶刪除 404、本人刪除 204、訊息隨 CASCADE 清空"

        await check("灌資料", "[1/7] 準備測試文件", t_setup)
        await check("第一輪", "[2/7] 第一輪提問並開新對話", t_first_turn)
        await check("追問", "[3/7] 追問時看得到前文", t_follow_up)
        await check("持久化", "[4/7] 訊息、來源、標題都有存", t_persisted)
        await check("租戶隔離", "[5/7] 對話的租戶隔離", t_isolation)
        await check("串流寫入", "[6/7] 串流回合也會寫進 DB", t_stream_persists)
        await check("刪除", "[7/7] 刪除與 CASCADE", t_delete)

    await cleanup()
    await db_shutdown()

    print(f"\n{LINE}")
    passed = sum(1 for _, g, _ in results if g)
    for name, good, msg in results:
        mark = "\033[32m✓\033[0m" if good else "\033[31m✗\033[0m"
        print(f" {mark} {name:10} {msg[:66]}")
    print(f"{LINE}")
    print(f" {passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
