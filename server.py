from fastapi import FastAPI
from fastapi.responses import FileResponse

INDEX_HTML = "index.html"

app = FastAPI(title="Agent Playground")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
