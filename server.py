from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from api import chat_router, conversations_router, documents_router
from database import db_shutdown, db_startup

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_startup()
    print("pool is prepared")

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
