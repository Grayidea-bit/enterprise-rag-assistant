"""Agent 的檢索工具與請求範圍相依物件。"""

from dataclasses import dataclass, field

from pydantic_ai import RunContext

from core.embedding import embed_query
from database.func import retrieve


@dataclass
class Retrieved:
    """一筆命中的 chunk。"""

    content: str
    distance: float
    title: str | None
    source: str


@dataclass
class RagDeps:
    """每次 agent.run() 傳入的請求範圍狀態。

    retrieved 是刻意設計的回傳通道:工具的字串回傳值是給模型看的,
    但端點也需要知道實際引用了哪些來源,所以工具順手記在這裡。
    """

    tenant_id: str
    limit: int = 5
    max_distance: float | None = None
    retrieved: list[Retrieved] = field(default_factory=list)

    def unique_sources(self) -> list[Retrieved]:
        """去掉重複命中(模型可能用不同措辭連續搜好幾次),保留距離最近的那筆。"""
        best: dict[tuple[str, str], Retrieved] = {}
        for hit in self.retrieved:
            key = (hit.source, hit.content)
            if key not in best or hit.distance < best[key].distance:
                best[key] = hit
        return sorted(best.values(), key=lambda h: h.distance)


async def search_knowledge_base(ctx: RunContext[RagDeps], query: str) -> str:
    """在企業知識庫中搜尋與問題相關的文件段落。

    回答任何有關內部文件、規章、流程的問題前都必須先呼叫這個工具,
    不要憑既有知識作答。

    Args:
        query: 要搜尋的問句或關鍵語句,使用與使用者相同的語言。
    """
    embedding = await embed_query(query)
    rows = await retrieve(
        ctx.deps.tenant_id,
        embedding,
        query,
        limit=ctx.deps.limit,
        max_distance=ctx.deps.max_distance,
    )
    if not rows:
        return "知識庫中找不到與這個問題相關的內容。"

    hits = [
        Retrieved(content=r[0], distance=float(r[1]), title=r[2], source=r[3])
        for r in rows
    ]
    ctx.deps.retrieved.extend(hits)

    blocks = []
    for i, hit in enumerate(hits, start=1):
        label = hit.source
        if hit.title and hit.title != hit.source:
            label = f"{hit.source}({hit.title})"
        blocks.append(f"[{i}] 來源:{label} 相似距離:{hit.distance:.4f}\n{hit.content}")
    return "\n\n".join(blocks)
