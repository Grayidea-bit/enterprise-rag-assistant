"""從上傳的檔案抽出純文字。

每種格式一個抽取器,失敗時丟 ExtractionError 並附上人看得懂的原因 ——
上傳失敗最令人挫折的就是只拿到「處理失敗」四個字。
"""

import io
from pathlib import PurePosixPath

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

# 抽出來的字數低於這個值就當成沒有文字層(掃描檔 / 純圖片 PDF)
MIN_PDF_CHARS = 20


class ExtractionError(Exception):
    """檔案無法轉成文字。訊息會直接回給呼叫端,所以要寫得具體。"""


def _extract_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ExtractionError(f"檔案不是有效的 UTF-8 文字:{e}") from e


def _extract_pdf(raw: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(raw))
    except PdfReadError as e:
        raise ExtractionError(f"PDF 檔案損毀或格式不正確:{e}") from e

    if reader.is_encrypted:
        # 有些 PDF 只設了空白擁有者密碼,試著解開;真的要密碼就放棄
        try:
            if not reader.decrypt(""):
                raise ExtractionError("PDF 有密碼保護,請先解除保護再上傳")
        except (PdfReadError, NotImplementedError) as e:
            raise ExtractionError(f"PDF 有密碼保護且無法解開:{e}") from e

    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            pages.append("")
            print(f"PDF 第 {i} 頁抽取失敗,已跳過:{type(e).__name__}: {e}")

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if len(text) < MIN_PDF_CHARS:
        raise ExtractionError(
            f"這份 PDF 共 {len(reader.pages)} 頁,但抽不出文字層 —— "
            "很可能是掃描檔或純圖片。本系統不做 OCR,請改上傳有文字層的版本。"
        )
    return text


def extract(filename: str, raw: bytes) -> str:
    """依副檔名抽取文字。不支援的格式與抽取失敗都丟 ExtractionError。"""
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _extract_text(raw)
    if suffix in PDF_SUFFIXES:
        return _extract_pdf(raw)
    raise ExtractionError(
        f"目前只支援 {sorted(SUPPORTED_SUFFIXES)},收到的是 '{suffix or '(無副檔名)'}'"
    )
