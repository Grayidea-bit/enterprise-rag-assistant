"""檢索問答 smoke test。

驗證 query 這半條路:向量檢索 → agent 呼叫檢索工具 → 帶引用的回答 → SSE 串流。
需要可連線的 pgvector 資料庫與 LLM 端點:

    python scripts/smoke_chat.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from config import env_settings  # noqa: E402
from database import db_shutdown, db_startup  # noqa: E402
from database.conn import pool  # noqa: E402
from server import app  # noqa: E402

LINE = "─" * 62
TENANT_A = "smoke-chat-a"
TENANT_B = "smoke-chat-b"  # 刻意留空,驗證「查不到就說查不到」

DOC = """# 差旅費用報支規定

國內出差每日膳雜費上限為新臺幣六百元,住宿費核實報支且上限為新臺幣二千八百元。
搭乘高鐵一律以標準車廂為準,商務車廂差額需由員工自行負擔。

## 報支期限

出差結束後十四個工作天內須完成報支,逾期需經單位主管專案簽核。
未附收據正本者一律不予核銷,電子發票需列印明細併附。

## 國外出差

國外出差的膳雜費依前往國家分級,甲級地區每日上限為美金一百二十元。
機票以經濟艙為原則,飛行時間超過十小時者得核准商務艙。
"""

results: list[tuple[str, bool, str]] = []


def ok(name: str, msg: str, t: float):
    print(f"  \033[32m✓\033[0m {msg}  \033[2m({t:.1f}s)\033[0m")
    results.append((name, True, msg))


def fail(name: str, err: str, t: float):
    print(f"  \033[31m✗\033[0m {err}  \033[2m({t:.1f}s)\033[0m")
    results.append((name, False, err))


async def check(name: str, title: str, fn):
    print(f"\n{title}")
    t = time.perf_counter()
    try:
        ok(name, await fn(), time.perf_counter() - t)
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}", time.perf_counter() - t)


async def cleanup():
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM documents WHERE tenant_id = ANY(%s)", ([TENANT_A, TENANT_B],)
        )


async def main() -> int:
    print(LINE)
    print(" 檢索問答 smoke test")
    print(LINE)
    print(f" chat      {env_settings.CHAT_MODEL}")
    print(f" embedding {env_settings.EMBEDDING_MODEL}")

    await db_startup()
    await cleanup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=600
    ) as client:
        a = {"X-Tenant-Id": TENANT_A}
        b = {"X-Tenant-Id": TENANT_B}

        async def t_setup():
            r = await client.post(
                "/documents",
                headers=a,
                files={"file": ("差旅規定.md", DOC.encode("utf-8"), "text/markdown")},
                data={"chunk_size": 200, "overlap": 40},
            )
            if r.status_code != 201:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return f"灌入 {r.json()['chunks']} 個 chunk"

        async def t_search():
            r = await client.post(
                "/search", headers=a, json={"query": "住宿費可以報多少?", "limit": 3}
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            hits = r.json()
            if not hits:
                raise RuntimeError("檢索不到任何結果")
            if any(h["source"] != "差旅規定.md" for h in hits):
                raise RuntimeError(f"撈到別的來源: {[h['source'] for h in hits]}")
            if hits != sorted(hits, key=lambda h: h["distance"]):
                raise RuntimeError("結果沒有依距離排序")
            empty = (await client.post("/search", headers=b, json={"query": "住宿費"})).json()
            if empty:
                raise RuntimeError(f"空租戶不該有結果: {empty}")
            return f"{len(hits)} 筆(最近 {hits[0]['distance']:.4f}),空租戶回空陣列"

        async def t_chat():
            r = await client.post(
                "/chat", headers=a, json={"question": "國內出差住宿費上限是多少?"}
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            body = r.json()
            answer = (body.get("answer") or "").strip()
            if not answer:
                raise RuntimeError("回答是空的")
            if not body.get("sources"):
                raise RuntimeError(f"沒有回傳來源,代表工具沒被呼叫:{answer[:120]}")
            if "二千八百" not in answer and "2800" not in answer and "2,800" not in answer:
                raise RuntimeError(f"答案沒引用到文件裡的數字:{answer[:150]}")
            return f"引用 {len(body['sources'])} 個來源,答案命中金額:{answer[:50]}…"

        async def t_no_data():
            r = await client.post(
                "/chat", headers=b, json={"question": "國內出差住宿費上限是多少?"}
            )
            body = r.json()
            answer = (body.get("answer") or "").strip()
            if body.get("sources"):
                raise RuntimeError(f"空租戶不該有來源: {body['sources']}")
            if "找不到" not in answer and "沒有" not in answer and "無法" not in answer:
                raise RuntimeError(f"空租戶應拒答,實際卻回答了:{answer[:150]}")
            return f"空租戶正確拒答:{answer[:44]}…"

        async def t_stream():
            events: list[tuple[str, dict]] = []
            async with client.stream(
                "POST", "/chat/stream", headers=a, json={"question": "報支期限是幾天?"}
            ) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                name = None
                async for line in r.aiter_lines():
                    if line.startswith("event: "):
                        name = line[7:]
                    elif line.startswith("data: ") and name:
                        events.append((name, json.loads(line[6:])))
            kinds = [n for n, _ in events]
            if "error" in kinds:
                raise RuntimeError(f"串流回報錯誤: {dict(events)['error']}")
            if not kinds or kinds[0] != "sources":
                raise RuntimeError(f"第一個事件應該是 sources,實際 {kinds[:3]}")
            if kinds[-1] != "done":
                raise RuntimeError(f"最後一個事件應該是 done,實際 {kinds[-3:]}")
            deltas = [p["text"] for n, p in events if n == "delta"]
            if len(deltas) < 2:
                raise RuntimeError(f"只收到 {len(deltas)} 個 delta,不算串流")
            text = "".join(deltas).strip()
            if not text:
                raise RuntimeError("串流出來的文字是空的")
            n_src = len(dict(events)["sources"]["sources"])
            return f"{len(deltas)} 個 delta / {len(text)} 字,先送 {n_src} 個來源:{text[:36]}…"

        async def t_validation():
            cases = [
                ({"question": ""}, 422, "空問題"),
                ({"question": "x", "limit": 0}, 422, "limit=0"),
                ({"question": "x", "limit": 999}, 422, "limit 過大"),
                ({"question": "x", "max_distance": -1}, 422, "負距離"),
            ]
            for payload, expected, label in cases:
                got = (await client.post("/chat", headers=a, json=payload)).status_code
                if got != expected:
                    raise RuntimeError(f"{label} 應回 {expected},實際 {got}")
            return "空問題 / limit 越界 / 負距離 皆回 422"

        await check("灌資料", "[1/6] 準備測試文件", t_setup)
        await check("向量檢索", "[2/6] POST /search 純檢索", t_search)
        await check("RAG 問答", "[3/6] POST /chat 帶引用回答", t_chat)
        await check("空租戶拒答", "[4/6] 查無資料時不編造", t_no_data)
        await check("SSE 串流", "[5/6] POST /chat/stream", t_stream)
        await check("參數驗證", "[6/6] 錯誤參數", t_validation)

    await cleanup()
    await db_shutdown()

    print(f"\n{LINE}")
    passed = sum(1 for _, good, _ in results if good)
    for name, good, msg in results:
        mark = "\033[32m✓\033[0m" if good else "\033[31m✗\033[0m"
        print(f" {mark} {name:12} {msg[:64]}")
    print(f"{LINE}")
    print(f" {passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
