"""檢索品質評估。

拿 eval/dataset.json 灌進一個專用租戶,對每一種檢索模式跑完所有問題,
算 recall@k 與 MRR,並依題型(字面 / 換句話說)分開看。

    python scripts/eval_retrieval.py                 # 比較 vector 與 hybrid
    python scripts/eval_retrieval.py --mode hybrid   # 只跑一種
    python scripts/eval_retrieval.py --k 1 3 5 10

命中的定義:top-k 之中,存在一段來自正確文件、且包含 must_contain 字串的段落。
用字串而不是 chunk id 當 ground truth,是為了讓評估集在改變切塊策略之後依然有效。
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import env_settings  # noqa: E402
from core.chunking import split_text  # noqa: E402
from core.embedding import embed_documents, embed_query  # noqa: E402
from database import db_shutdown, db_startup  # noqa: E402
from database.conn import pool  # noqa: E402
from database.func import (  # noqa: E402
    delete_chunks,
    insert_chunks,
    retrieve,
    upsert_document,
)

TENANT = "eval-retrieval"
DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset.json"
LINE = "─" * 72


async def ingest(documents: list[dict], chunk_size: int, overlap: int) -> int:
    total = 0
    for doc in documents:
        contents = split_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        embeddings = await embed_documents(contents)
        document_id, existed = await upsert_document(
            TENANT, doc["title"], doc["source"], None
        )
        if existed:
            await delete_chunks(TENANT, document_id)
        total += await insert_chunks(TENANT, document_id, contents, embeddings)
    return total


def rank_of_hit(rows, question: dict) -> int | None:
    """回傳第一個命中的名次(1-based),沒命中回 None。"""
    for i, (content, _distance, _title, source) in enumerate(rows, start=1):
        if source == question["source"] and question["must_contain"] in content:
            return i
    return None


async def run_mode(mode: str, questions: list[dict], ks: list[int]) -> dict:
    top_k = max(ks)
    ranks: list[int | None] = []
    elapsed = 0.0
    for q in questions:
        embedding = await embed_query(q["q"])
        t = time.perf_counter()
        rows = await retrieve(TENANT, embedding, q["q"], limit=top_k, mode=mode)
        elapsed += time.perf_counter() - t
        ranks.append(rank_of_hit(rows, q))

    def recall_at(k: int, subset: list[int | None]) -> float:
        if not subset:
            return 0.0
        return sum(1 for r in subset if r is not None and r <= k) / len(subset)

    def mrr(subset: list[int | None]) -> float:
        if not subset:
            return 0.0
        return sum(1 / r for r in subset if r is not None) / len(subset)

    by_kind: dict[str, list[int | None]] = {}
    for q, r in zip(questions, ranks):
        by_kind.setdefault(q["kind"], []).append(r)

    return {
        "recall": {k: recall_at(k, ranks) for k in ks},
        "mrr": mrr(ranks),
        "ms": elapsed / len(questions) * 1000,
        "by_kind": {
            kind: {"recall": recall_at(max(ks), rs), "mrr": mrr(rs), "n": len(rs)}
            for kind, rs in by_kind.items()
        },
        "misses": [q["q"] for q, r in zip(questions, ranks) if r is None],
        "ranks": ranks,
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="檢索品質評估")
    ap.add_argument("--mode", choices=["vector", "hybrid"], help="只跑單一模式")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], help="recall@k 的 k")
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=60)
    args = ap.parse_args()

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    documents, questions = data["documents"], data["questions"]
    modes = [args.mode] if args.mode else ["vector", "hybrid"]
    ks = sorted(args.k)

    print(LINE)
    print(" 檢索品質評估")
    print(LINE)
    print(f" embedding {env_settings.EMBEDDING_MODEL} ({env_settings.EMBEDDING_DIM} 維)")
    print(f" 資料集    {len(documents)} 份文件 / {len(questions)} 題")
    print(f" 切塊      chunk_size={args.chunk_size} overlap={args.overlap}")

    await db_startup()
    chunks = await ingest(documents, args.chunk_size, args.overlap)
    print(f" 已灌入    {chunks} 個 chunk")

    results = {}
    for mode in modes:
        print(f"\n 跑 {mode} …", end="", flush=True)
        results[mode] = await run_mode(mode, questions, ks)
        print(" 完成")

    # ── 總表 ──
    head = f"{'模式':<10}" + "".join(f"{'recall@'+str(k):>11}" for k in ks)
    head += f"{'MRR':>9}{'ms/題':>9}"
    print(f"\n{LINE}\n{head}\n{'-' * 72}")
    for mode in modes:
        r = results[mode]
        row = f"{mode:<10}" + "".join(f"{r['recall'][k]:>10.1%} " for k in ks)
        row += f"{r['mrr']:>8.3f} {r['ms']:>8.0f}"
        print(row)

    # ── 依題型拆開:hybrid 的價值應該集中在字面題 ──
    kinds = sorted({q["kind"] for q in questions})
    print(f"\n{'依題型 (recall@' + str(max(ks)) + ')':<24}" + "".join(f"{m:>16}" for m in modes))
    print("-" * 72)
    for kind in kinds:
        n = results[modes[0]]["by_kind"][kind]["n"]
        row = f"{kind + f' (n={n})':<24}"
        for mode in modes:
            row += f"{results[mode]['by_kind'][kind]['recall']:>15.1%} "
        print(row)

    # ── 差異 ──
    if len(modes) == 2:
        a, b = modes
        print(f"\n{LINE}")
        for k in ks:
            d = results[b]['recall'][k] - results[a]['recall'][k]
            sign = "+" if d >= 0 else ""
            print(f" recall@{k}: {a} {results[a]['recall'][k]:.1%} → {b} {results[b]['recall'][k]:.1%}  ({sign}{d:.1%})")
        moved = [
            (q["q"], results[a]["ranks"][i], results[b]["ranks"][i])
            for i, q in enumerate(questions)
            if results[a]["ranks"][i] != results[b]["ranks"][i]
        ]
        if moved:
            print(f"\n 名次有變動的題目({len(moved)} 題):")
            for q, ra, rb in moved[:12]:
                fmt = lambda r: "未命中" if r is None else f"#{r}"
                print(f"   {fmt(ra):>6} → {fmt(rb):<6}  {q}")

    for mode in modes:
        if results[mode]["misses"]:
            print(f"\n {mode} 完全沒撈到的題目:")
            for q in results[mode]["misses"]:
                print(f"   · {q}")

    async with pool.connection() as conn:
        await conn.execute("DELETE FROM documents WHERE tenant_id = %s", (TENANT,))
    await db_shutdown()
    print(f"\n{LINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
