"""上傳管線的整合行為。"""

import pytest

from database.conn import pool
from tests.integration.conftest import upload
from tests.pdf_fixture import make_pdf

pytestmark = pytest.mark.integration

LONG_DOC = "".join(f"第{i}條規定的內容說明。" for i in range(1, 60))


async def test_upload_returns_document_and_chunks(client, alice):
    response = await upload(client, alice, "policy.md", LONG_DOC, chunk_size=200, overlap=40)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["tenant_id"] == alice.id
    assert body["source"] == "policy.md"
    assert body["chunks"] > 1
    assert body["replaced"] is False


async def test_chunk_index_is_contiguous(client, alice):
    body = (await upload(client, alice, "policy.md", LONG_DOC, chunk_size=200, overlap=40)).json()
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT chunk_index FROM chunks WHERE document_id = %s ORDER BY chunk_index",
                (body["document_id"],),
            )
        ).fetchall()
    assert [r[0] for r in rows] == list(range(body["chunks"]))


async def test_reupload_replaces_instead_of_duplicating(client, alice):
    first = (await upload(client, alice, "policy.md", LONG_DOC, chunk_size=200)).json()
    second = (await upload(client, alice, "policy.md", LONG_DOC + "補充。", chunk_size=400)).json()

    assert second["replaced"] is True
    assert second["document_id"] == first["document_id"]

    listing = (await client.get("/documents", headers=alice.headers)).json()
    assert len(listing) == 1
    assert listing[0]["chunk_count"] == second["chunks"]


async def test_documents_are_scoped_to_the_tenant(client, alice, bob):
    await upload(client, alice, "alice.md", "alice 的機密文件內容。")
    await upload(client, bob, "bob.md", "bob 的機密文件內容。")

    seen_by_alice = (await client.get("/documents", headers=alice.headers)).json()
    seen_by_bob = (await client.get("/documents", headers=bob.headers)).json()
    assert [d["source"] for d in seen_by_alice] == ["alice.md"]
    assert [d["source"] for d in seen_by_bob] == ["bob.md"]


class TestRejections:
    async def test_unsupported_extension(self, client, alice):
        assert (await upload(client, alice, "report.docx", "x")).status_code == 415

    async def test_invalid_utf8(self, client, alice):
        response = await client.post(
            "/documents",
            headers=alice.headers,
            files={"file": ("bad.txt", b"\xff\xfe\x00binary", "text/plain")},
        )
        assert response.status_code == 400

    async def test_blank_document(self, client, alice):
        assert (await upload(client, alice, "empty.txt", "   \n\n ")).status_code == 400

    @pytest.mark.parametrize("form", [{"overlap": 99999}, {"chunk_size": 0}])
    async def test_invalid_chunking_parameters(self, client, alice, form):
        assert (await upload(client, alice, "a.txt", "內容", **form)).status_code == 422


class TestPdf:
    async def test_pdf_with_text_layer_is_ingested(self, client, alice):
        pdf = make_pdf(
            [
                "Procurement Policy",
                "Purchases above NTD 300000 require three written quotes.",
            ]
        )
        response = await client.post(
            "/documents",
            headers=alice.headers,
            files={"file": ("policy.pdf", pdf, "application/pdf")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["chunks"] >= 1

    async def test_scanned_pdf_is_rejected_with_an_explanation(self, client, alice):
        response = await client.post(
            "/documents",
            headers=alice.headers,
            files={"file": ("scan.pdf", make_pdf([], with_text=False), "application/pdf")},
        )
        assert response.status_code == 400
        assert "OCR" in response.json()["detail"]
