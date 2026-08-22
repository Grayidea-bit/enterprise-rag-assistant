"""密閉整合測試的固定裝置。

需要一個可連線的 PostgreSQL(CI 用 service container 提供),但**不需要**
任何 LLM 端點 —— 模型換成 pydantic-ai 內建的測試替身。

代價是排序品質不可驗:TestEmbeddingModel 對任何輸入都回傳全 1.0 的向量,
所有段落的餘弦距離都是 0。這裡驗的是「管線接得對不對」,
「答得好不好」交給 scripts/smoke_*.py 打真實端點。
"""

import os
from collections.abc import AsyncIterator

import httpx
import psycopg
import pytest
import pytest_asyncio
from pydantic_ai.embeddings.test import TestEmbeddingModel
from pydantic_ai.models.test import TestModel

from config import env_settings
from core.agent import get_agent
from core.auth import generate_key
from core.embedding import get_embedder
from database import db_shutdown, db_startup
from database.conn import pool
from database.func import insert_api_key
from database.migrate import upgrade
from server import app


def _database_reachable() -> bool:
    try:
        with psycopg.connect(env_settings.DATABASE_URL, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _database() -> AsyncIterator[None]:
    if not _database_reachable():
        pytest.skip(
            f"連不到資料庫 {env_settings.DATABASE_URL.rsplit('@', 1)[-1]};"
            "整合測試需要 PostgreSQL(docker compose up -d db)",
            allow_module_level=True,
        )
    await db_startup()
    await upgrade()
    yield
    await db_shutdown()


@pytest.fixture(autouse=True)
def _stubbed_models():
    """把 chat 與 embedding 換成測試替身,整套測試因此不需要外部服務。"""
    stub_embeddings = TestEmbeddingModel(dimensions=env_settings.EMBEDDING_DIM)
    with (
        get_agent().override(model=TestModel()),
        get_embedder().override(model=stub_embeddings),
    ):
        yield


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=60
    ) as http_client:
        yield http_client


class Tenant:
    """一個測試租戶:自己的 id、自己的金鑰 headers。"""

    def __init__(self, tenant_id: str, headers: dict[str, str]):
        self.id = tenant_id
        self.headers = headers


async def _make_tenant(label: str) -> Tenant:
    # 用 pid 讓平行執行的 worker 不會互相踩到
    tenant_id = f"it-{label}-{os.getpid()}"
    key, key_hash, prefix = generate_key()
    await insert_api_key(key_hash, tenant_id, "integration-test", prefix)
    return Tenant(tenant_id, {"Authorization": f"Bearer {key}"})


async def _purge(*tenant_ids: str) -> None:
    async with pool.connection() as conn:
        for table in ("conversations", "documents", "api_keys"):
            await conn.execute(
                f"DELETE FROM {table} WHERE tenant_id = ANY(%s)", (list(tenant_ids),)
            )


@pytest_asyncio.fixture
async def alice() -> AsyncIterator[Tenant]:
    tenant = await _make_tenant("alice")
    yield tenant
    await _purge(tenant.id)


@pytest_asyncio.fixture
async def bob() -> AsyncIterator[Tenant]:
    tenant = await _make_tenant("bob")
    yield tenant
    await _purge(tenant.id)


def upload(client: httpx.AsyncClient, tenant: Tenant, name: str, body: str, **form):
    return client.post(
        "/documents",
        headers=tenant.headers,
        files={"file": (name, body.encode("utf-8"), "text/markdown")},
        data=form,
    )
