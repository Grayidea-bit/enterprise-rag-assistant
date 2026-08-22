"""產生最小可用的測試 PDF,不引入 reportlab 之類的依賴。

只支援 ASCII 與 Helvetica —— 測試要驗的是抽取邏輯,不是排版。
"""


def make_pdf(lines: list[str], *, with_text: bool = True) -> bytes:
    """組出一份單頁 PDF。with_text=False 會產生沒有文字層的頁面(模擬掃描檔)。"""
    if with_text:
        body = "BT /F1 12 Tf 50 750 Td 14 TL\n"
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            body += f"({escaped}) Tj T*\n"
        body += "ET"
    else:
        body = "0 0 0 rg 50 700 200 50 re f"  # 只有一個黑色方塊,沒有文字

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(body)} >>\nstream\n{body}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)
