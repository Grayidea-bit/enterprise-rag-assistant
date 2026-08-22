"""Ingest 管線 smoke test。

驗證 上傳 → 切塊 → embed → 寫入 pgvector → 租戶隔離檢索 這條路真的通。
需要一個可連線的 pgvector 資料庫:

    DATABASE_URL=postgresql://postgres:secret@localhost:5433/enterprise_rag \
        python scripts/smoke_ingest.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config import env_settings
from database import db_shutdown, db_startup
from database.conn import pool
from database.func import search_chunks
from scripts._smoke_auth import drop_keys, mint
from server import app
from tests.pdf_fixture import make_pdf

LINE = "─" * 62
TENANT_A = "smoke-tenant-a"
TENANT_B = "smoke-tenant-b"

# 刻意寫長一點,才會切出多個 chunk,批次寫入與 chunk_index 排序才真的被驗到
DOC_A = """# 理賠作業手冊

申請醫療理賠時,必須備齊診斷證明書正本、費用收據正本與病歷摘要。
診斷證明書需由主治醫師開立,並載明疾病名稱與治療期間,缺一不可。
費用收據須為正本並蓋有醫療院所收據章,影本一律不予受理。

## 審核時程

文件齊全後,一般案件的審核時程為五個工作天。
若需要補件,審核時程會從補件完成日重新計算,並以書面通知要保人。
重大傷病或需要再保人覆核的案件,審核時程得延長至十五個工作天。
逾期未完成審核者,應主動通知要保人並說明延遲原因與預計完成日。

## 給付上限

住院日額給付每日上限為新臺幣三千元,同一保單年度累計不得超過一百八十日。
手術給付依手術等級表計算,最高不得超過保險金額的百分之二十。
同一事故合併申請多項給付時,採擇優給付原則,不重複計算。

## 除外責任

要保人或被保險人的故意行為所致者,本公司不負給付責任。
被保險人犯罪行為或拒捕過程中所致的傷害,亦屬除外責任範圍。
非因治療必要的整型手術、美容手術與例行健康檢查,均不在給付範圍內。

## 申訴管道

對理賠結果有異議者,得於接獲通知後六十日內以書面提出申訴。
申訴案件由獨立於原承辦單位的爭議處理小組重新審視,並於三十日內回覆。
"""

DOC_B = """# 出貨流程規範

倉庫收到訂單後,揀貨人員需在兩小時內完成揀貨並列印出貨單。
出貨單一式三聯,分別留存於倉庫、財務與客戶。

## 品檢

每批出貨前需抽驗百分之五的品項,發現瑕疵即整批退回產線重工。

## 運送

一般件由合作物流於次日配送,冷鏈商品需使用專車並全程記錄溫度。
"""

results: list[tuple[str, bool, str]] = []


def ok(name: str, msg: str, elapsed: float):
    print(f"  \033[32m✓\033[0m {msg}  \033[2m({elapsed:.1f}s)\033[0m")
    results.append((name, True, msg))


def fail(name: str, err: str, elapsed: float):
    print(f"  \033[31m✗\033[0m {err}  \033[2m({elapsed:.1f}s)\033[0m")
    results.append((name, False, err))


async def check(name: str, title: str, fn):
    print(f"\n{title}")
    t = time.perf_counter()
    try:
        msg = await fn()
        ok(name, msg, time.perf_counter() - t)
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}", time.perf_counter() - t)


async def cleanup():
    await drop_keys(TENANT_A, TENANT_B)
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM documents WHERE tenant_id = ANY(%s)", ([TENANT_A, TENANT_B],)
        )


def upload(client: httpx.AsyncClient, hdr: dict, name: str, body: str, **data):
    return client.post(
        "/documents",
        headers=hdr,
        files={"file": (name, body.encode("utf-8"), "text/markdown")},
        data=data,
    )


async def main() -> int:
    print(LINE)
    print(" Ingest 管線 smoke test")
    print(LINE)
    print(f" DB       {env_settings.DATABASE_URL.rsplit('@', 1)[-1]}")
    print(f" embedding {env_settings.EMBEDDING_MODEL} ({env_settings.EMBEDDING_DIM} 維)")

    await db_startup()
    await cleanup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        state: dict = {}
        HA = await mint(TENANT_A)
        HB = await mint(TENANT_B)

        async def t_upload():
            # 順便驗 Form 參數:縮小 chunk_size 逼出多塊,批次寫入才真的被走到
            r = await upload(client, HA, "理賠手冊.md", DOC_A, chunk_size=200, overlap=40)
            if r.status_code != 201:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            body = r.json()
            state["doc_a"] = body["document_id"]
            if body["chunks"] < 2:
                raise RuntimeError(f"這份文件應該切出多個 chunk,實際 {body['chunks']}")
            if body["replaced"]:
                raise RuntimeError("首次上傳不該是 replaced")
            # 確認批次寫入的 chunk_index 是 0..n-1 且沒有跳號
            async with pool.connection() as conn:
                rows = await (
                    await conn.execute(
                        "SELECT chunk_index FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                        (body["document_id"],),
                    )
                ).fetchall()
            indexes = [r[0] for r in rows]
            if indexes != list(range(body["chunks"])):
                raise RuntimeError(f"chunk_index 不連續: {indexes}")
            return f"document_id={body['document_id']} chunks={body['chunks']} index 連續"

        async def t_upload_b():
            r = await upload(client, HB, "出貨流程.md", DOC_B)
            if r.status_code != 201:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return f"tenant B document_id={r.json()['document_id']}"

        async def t_replace():
            r = await upload(client, HA, "理賠手冊.md", DOC_A + "\n補充條款。\n")
            body = r.json()
            if not body["replaced"]:
                raise RuntimeError("重複上傳同一 source 應該回 replaced=True")
            if body["document_id"] != state["doc_a"]:
                raise RuntimeError("重複上傳不該產生新的 document_id")
            listing = (await client.get("/documents", headers=HA)).json()
            if len(listing) != 1:
                raise RuntimeError(f"同一 source 上傳兩次卻有 {len(listing)} 份文件")
            return f"覆蓋成功,仍是 1 份文件 / {listing[0]['chunk_count']} chunks"

        async def t_list_isolation():
            a = (await client.get("/documents", headers=HA)).json()
            b = (await client.get("/documents", headers=HB)).json()
            if {d["source"] for d in a} != {"理賠手冊.md"}:
                raise RuntimeError(f"租戶 A 看到了不該看到的文件: {[d['source'] for d in a]}")
            if {d["source"] for d in b} != {"出貨流程.md"}:
                raise RuntimeError(f"租戶 B 看到了不該看到的文件: {[d['source'] for d in b]}")
            return "A 只看到自己的 1 份,B 只看到自己的 1 份"

        async def t_search_isolation():
            from core.embedding import embed_query

            vec = await embed_query("理賠需要準備哪些文件?")
            hits_a = await search_chunks(TENANT_A, vec, limit=3)
            hits_b = await search_chunks(TENANT_B, vec, limit=3)
            if not hits_a:
                raise RuntimeError("租戶 A 查不到任何結果")
            if any(h[3] != "理賠手冊.md" for h in hits_a):
                raise RuntimeError(f"租戶 A 撈到別人的文件: {[h[3] for h in hits_a]}")
            if any(h[3] != "出貨流程.md" for h in hits_b):
                raise RuntimeError(f"租戶 B 撈到別人的文件: {[h[3] for h in hits_b]}")
            top = hits_a[0]
            return f"A 命中 {len(hits_a)} 筆(最近 {top[1]:.4f}),B 完全隔離"

        async def t_distance_threshold():
            from core.embedding import embed_query

            vec = await embed_query("量子色動力學的漸近自由")
            loose = await search_chunks(TENANT_A, vec, limit=3)
            strict = await search_chunks(TENANT_A, vec, limit=3, max_distance=0.3)
            if not loose:
                raise RuntimeError("無門檻時應該還是會回傳結果")
            if strict:
                raise RuntimeError(f"門檻 0.3 不該讓不相關內容通過: {strict[0][1]:.4f}")
            return f"無門檻回 {len(loose)} 筆(最近 {loose[0][1]:.4f}),加門檻後正確擋掉"

        async def t_rejects():
            bad_ext = await upload(client, HA, "報表.docx", "x")
            if bad_ext.status_code != 415:
                raise RuntimeError(f"副檔名不符應回 415,實際 {bad_ext.status_code}")
            bad_utf8 = await client.post(
                "/documents",
                headers=HA,
                files={"file": ("壞檔.txt", b"\xff\xfe\x00binary", "text/plain")},
            )
            if bad_utf8.status_code != 400:
                raise RuntimeError(f"非 UTF-8 應回 400,實際 {bad_utf8.status_code}")
            empty = await upload(client, HA, "空的.txt", "   \n\n  ")
            if empty.status_code != 400:
                raise RuntimeError(f"空內容應回 400,實際 {empty.status_code}")
            bad_param = await upload(client, HA, "x.txt", "hello", overlap=9999)
            if bad_param.status_code != 422:
                raise RuntimeError(f"overlap 超界應回 422,實際 {bad_param.status_code}")
            return "415 / 400 / 400 / 422 皆正確擋下"

        await check("上傳 A", "[1/8] 上傳文件（租戶 A）", t_upload)
        await check("上傳 B", "[2/8] 上傳文件（租戶 B）", t_upload_b)
        await check("重複上傳", "[3/8] 同名重傳走 upsert 而非長第二份", t_replace)
        await check("列表隔離", "[4/8] GET /documents 的租戶隔離", t_list_isolation)
        await check("檢索隔離", "[5/8] 向量檢索的租戶隔離", t_search_isolation)
        await check("距離門檻", "[6/8] max_distance 擋掉不相關結果", t_distance_threshold)

        async def t_pdf():
            pdf = make_pdf(
                [
                    "Procurement Policy",
                    "Purchases above NTD 300000 require three written quotes.",
                    "Acceptance must be completed within seven working days.",
                ]
            )
            r = await client.post(
                "/documents",
                headers=HA,
                files={"file": ("procurement.pdf", pdf, "application/pdf")},
                data={"chunk_size": 200, "overlap": 40},
            )
            if r.status_code != 201:
                raise RuntimeError(f"PDF 上傳失敗 HTTP {r.status_code}: {r.text[:200]}")
            if r.json()["chunks"] < 1:
                raise RuntimeError("PDF 沒有切出任何 chunk")
            # 掃描檔(沒有文字層)必須被擋下並說清楚原因
            scan = await client.post(
                "/documents",
                headers=HA,
                files={"file": ("scan.pdf", make_pdf([], with_text=False), "application/pdf")},
            )
            if scan.status_code != 400 or "OCR" not in scan.json()["detail"]:
                raise RuntimeError(f"掃描檔應回 400 並提及 OCR,實際 {scan.status_code}")
            return f"PDF 抽出 {r.json()['chunks']} 個 chunk,掃描檔回 400 並說明不做 OCR"

        await check("錯誤處理", "[7/8] 錯誤輸入的處理", t_rejects)
        await check("PDF", "[8/8] PDF 抽取與掃描檔處理", t_pdf)

    await cleanup()
    await db_shutdown()

    print(f"\n{LINE}")
    passed = sum(1 for _, good, _ in results if good)
    for name, good, msg in results:
        mark = "\033[32m✓\033[0m" if good else "\033[31m✗\033[0m"
        print(f" {mark} {name:10} {msg[:66]}")
    print(f"{LINE}")
    print(f" {passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
