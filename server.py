from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse

from api import chat_router, conversations_router, documents_router
from api.deps import resolve_tenant
from config import env_settings
from database import db_shutdown, db_startup

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_startup()
    print("pool is prepared")
    if env_settings.AUTH_MODE == "disabled":
        print(
            "⚠  AUTH_MODE=disabled — X-Tenant-Id 未經驗證即被採信。"
            "僅限本機開發,絕不可用於對外環境。"
        )

    yield

    await db_shutdown()
    print("pool is shutdown")


app = FastAPI(lifespan=lifespan)
app.include_router(documents_router)
app.include_router(conversations_router)
app.include_router(chat_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/me", tags=["auth"])
async def me(tenant_id: str = Depends(resolve_tenant)) -> dict[str, str]:
    """回報這個請求被解析成哪個租戶。前端用它確認金鑰有效。"""
    return {"tenant_id": tenant_id, "auth_mode": env_settings.AUTH_MODE}
