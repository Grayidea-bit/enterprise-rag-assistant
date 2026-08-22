"""文字切塊。

刻意不引入 tokenizer:bge-m3 吃得下 8192 token,以字元計算已經綽綽有餘,
為了精準算 token 而拉進 transformers 對這個專案太重。
中文約 1 字元 ≈ 1 token,英文更少,所以 800 字元離上限還很遠。
"""

from collections.abc import Sequence

# 由粗到細:先在段落切,不夠再句子,最後才硬切
SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ". ",
    "! ",
    "? ",
    "; ",
    " ",
    "",
)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100


def _split_keep(text: str, separator: str) -> list[str]:
    """依 separator 切開,但把 separator 留在前一段尾巴,避免標點被吃掉。"""
    parts = text.split(separator)
    pieces = [p + separator for p in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])
    return pieces


def _split_recursive(text: str, separators: Sequence[str], chunk_size: int) -> list[str]:
    """遞迴切到每一小段都不超過 chunk_size。"""
    if len(text) <= chunk_size:
        return [text] if text else []

    for i, separator in enumerate(separators):
        if separator == "":
            # 最後手段:沒有任何可用的分隔符,直接硬切
            return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]
        if separator not in text:
            continue

        pieces: list[str] = []
        for part in _split_keep(text, separator):
            if len(part) <= chunk_size:
                if part:
                    pieces.append(part)
            else:
                pieces.extend(_split_recursive(part, separators[i + 1 :], chunk_size))
        return pieces

    return [text]


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """把長文切成重疊的區塊。

    overlap 讓跨區塊的句子還有機會被檢索到 —— 答案剛好落在切點上是常見的失敗模式。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必須大於 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap 必須介於 0 與 chunk_size({chunk_size}) 之間")

    text = text.strip()
    if not text:
        return []

    pieces = _split_recursive(text, SEPARATORS, chunk_size)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) > chunk_size:
            chunks.append(buffer.strip())
            carry = buffer[-overlap:] if overlap else ""
            # 帶太多會讓下一塊一開始就爆掉,那就不帶
            buffer = (carry + piece) if len(carry) + len(piece) <= chunk_size else piece
        else:
            buffer += piece

    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks
