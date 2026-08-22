"""LLM 連線 smoke test。

驗證 .env 指向的 OpenAI 相容端點是否真的能用:設定解析 → embedding → chat → tool calling。
四項全綠才代表下一刀（ingest / 檢索工具）可以放心往上疊。

用法:
    python scripts/smoke_llm.py                  # 用 .env 的設定
    python scripts/smoke_llm.py --model qwen3-vl:8b   # 臨時換小模型測,冷啟動比較快
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser(description="LLM 連線 smoke test")
parser.add_argument("--model", help="臨時覆寫 CHAT_MODEL")
args = parser.parse_args()

# config 是 import 時就實體化的,覆寫必須趕在 import 之前
if args.model:
    os.environ["CHAT_MODEL"] = args.model

from config import env_settings  # noqa: E402
from core.embedding import embed_documents, embed_query  # noqa: E402
from core.llm import get_model  # noqa: E402

LINE = "─" * 62
results: list[tuple[str, bool, str]] = []


def mask(key: str | None) -> str:
    if not key:
        return "(未設定)"
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "****"


def step(n: int, title: str):
    print(f"\n[{n}/4] {title}")


def ok(msg: str, elapsed: float, name: str):
    print(f"  \033[32m✓\033[0m {msg}  \033[2m({elapsed:.1f}s)\033[0m")
    results.append((name, True, msg))


def fail(err: Exception, elapsed: float, name: str):
    print(f"  \033[31m✗\033[0m {type(err).__name__}: {err}  \033[2m({elapsed:.1f}s)\033[0m")
    results.append((name, False, f"{type(err).__name__}: {err}"))


async def check_config() -> None:
    step(1, "設定解析")
    t = time.perf_counter()
    try:
        chat_url, chat_key = env_settings.chat_target
        emb_url, emb_key = env_settings.embedding_target
        shared = "（沿用 chat）" if env_settings.embedding_endpoint_is_shared else "（獨立端點）"
        print(f"  chat       {chat_url}")
        print(f"             model={env_settings.CHAT_MODEL}  key={mask(chat_key)}")
        print(f"  embedding  {emb_url} {shared}")
        print(f"             model={env_settings.EMBEDDING_MODEL}  key={mask(emb_key)}  dim={env_settings.EMBEDDING_DIM}")
        if not chat_url.rstrip("/").endswith("/v1"):
            print("  \033[33m!\033[0m base_url 未以 /v1 結尾,多數 OpenAI 相容端點會 404")
        ok("設定可解析", time.perf_counter() - t, "設定")
    except Exception as e:
        fail(e, time.perf_counter() - t, "設定")


async def check_embedding() -> None:
    step(2, f"Embedding（{env_settings.EMBEDDING_MODEL}）")
    t = time.perf_counter()
    try:
        vec = await embed_query("企業知識庫的權限該怎麼設計?")
        batch = await embed_documents(["第一段文字", "第二段文字"])
        ok(f"query {len(vec)} 維 / batch {len(batch)} 筆", time.perf_counter() - t, "Embedding")
    except Exception as e:
        fail(e, time.perf_counter() - t, "Embedding")


async def check_chat() -> None:
    from pydantic_ai import Agent

    step(3, f"Chat（{env_settings.CHAT_MODEL}）")
    print("  \033[2m冷啟動需載入模型,大模型可能要數分鐘…\033[0m")
    t = time.perf_counter()
    try:
        agent = Agent(get_model(), instructions="用一句話回答,不要展開。")
        result = await agent.run("用一句話說明什麼是 RAG")
        text = (result.output or "").strip()
        if not text:
            raise RuntimeError("回覆為空字串 — 檢查 profile 的 openai_chat_thinking_field 設定")
        ok(f"回覆 {len(text)} 字:{text[:60]}…", time.perf_counter() - t, "Chat")
    except Exception as e:
        fail(e, time.perf_counter() - t, "Chat")


async def check_tool_calling() -> None:
    from pydantic_ai import Agent

    step(4, "Tool calling（驗證 strict profile 覆寫）")
    t = time.perf_counter()
    called = {"hit": False}

    def get_stock_price(symbol: str) -> str:
        """查詢股票的即時價格。"""
        called["hit"] = True
        return f"{symbol} 現價 123.45 元"

    try:
        agent = Agent(
            get_model(),
            instructions="需要外部資料時務必使用提供的工具,不要自己編造。",
            tools=[get_stock_price],
        )
        result = await agent.run("TSLA 現在多少錢?")
        if not called["hit"]:
            raise RuntimeError(
                f"模型沒有呼叫工具,直接回答了:{(result.output or '')[:80]}"
            )
        ok("工具確實被呼叫", time.perf_counter() - t, "Tool calling")
    except Exception as e:
        fail(e, time.perf_counter() - t, "Tool calling")


async def main() -> int:
    print(LINE)
    print(" LLM 連線 smoke test")
    print(LINE)

    await check_config()
    await check_embedding()
    await check_chat()
    await check_tool_calling()

    print(f"\n{LINE}")
    passed = sum(1 for _, good, _ in results if good)
    for name, good, msg in results:
        mark = "\033[32m✓\033[0m" if good else "\033[31m✗\033[0m"
        print(f" {mark} {name:14} {msg[:70]}")
    print(f"{LINE}")
    print(f" {passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
