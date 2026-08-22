"""core/extract.py 的單元測試。"""

import pytest

from core.extract import MIN_PDF_CHARS, ExtractionError, extract
from tests.pdf_fixture import make_pdf


class TestText:
    @pytest.mark.parametrize("name", ["a.txt", "a.md", "a.markdown", "a.text", "A.MD"])
    def test_text_suffixes(self, name):
        assert extract(name, "內容".encode()) == "內容"

    def test_non_utf8_is_rejected_with_a_readable_reason(self):
        with pytest.raises(ExtractionError, match="UTF-8"):
            extract("a.txt", b"\xff\xfe\x00binary")

    def test_utf8_bom_and_newlines_survive(self):
        assert extract("a.md", "# 標題\n\n內文\n".encode()) == "# 標題\n\n內文\n"


class TestPdf:
    def test_extracts_text_layer(self):
        text = extract("a.pdf", make_pdf(["Lodging capped at NTD 2800", "Meals 600"]))
        assert "2800" in text and "Meals" in text

    def test_multiple_lines_are_preserved(self):
        # 每行都要夠長,總長度得超過 MIN_PDF_CHARS 才不會被當成掃描檔
        lines = ["alpha section heading", "beta paragraph body", "gamma closing note"]
        text = extract("a.pdf", make_pdf(lines))
        for word in ("alpha", "beta", "gamma"):
            assert word in text

    def test_scanned_pdf_without_text_layer_is_rejected(self):
        with pytest.raises(ExtractionError, match="OCR"):
            extract("scan.pdf", make_pdf([], with_text=False))

    def test_almost_empty_text_layer_is_treated_as_scanned(self):
        with pytest.raises(ExtractionError, match="OCR"):
            extract("thin.pdf", make_pdf(["ab"]))

    def test_enough_text_passes_the_threshold(self):
        text = extract("ok.pdf", make_pdf(["x" * (MIN_PDF_CHARS + 5)]))
        assert len(text) >= MIN_PDF_CHARS

    def test_corrupt_pdf_is_rejected(self):
        with pytest.raises(ExtractionError, match=r"損毀|格式"):
            extract("broken.pdf", b"%PDF-1.4\nnot really a pdf")


class TestDispatch:
    @pytest.mark.parametrize("name", ["a.docx", "a.xlsx", "a.png", "noextension"])
    def test_unsupported_suffixes(self, name):
        with pytest.raises(ExtractionError, match="只支援"):
            extract(name, b"x")

    def test_suffix_is_case_insensitive(self):
        assert extract("A.TXT", b"hi") == "hi"

    def test_error_names_the_offending_suffix(self):
        with pytest.raises(ExtractionError, match=r"\.docx"):
            extract("report.docx", b"x")
