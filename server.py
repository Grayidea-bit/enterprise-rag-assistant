from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from database import db_shutdown, db_startup

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_startup()
    print("pool is prepared")

    yield

    db_shutdown()
    print("pool is shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
