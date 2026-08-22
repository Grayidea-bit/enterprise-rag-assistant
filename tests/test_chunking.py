"""core/chunking.py 的單元測試。純函式,不需要資料庫或 LLM。"""

import pytest

from core.chunking import DEFAULT_CHUNK_SIZE, split_text


@pytest.mark.parametrize("text", ["", "   ", "\n\n  \t \n"])
def test_blank_input_yields_nothing(text):
    assert split_text(text) == []


def test_short_text_stays_one_chunk():
    assert split_text("這是一段很短的文字。") == ["這是一段很短的文字。"]


def test_no_chunk_exceeds_chunk_size():
    text = "第一句話。" * 400
    for chunk in split_text(text, chunk_size=200, overlap=40):
        assert len(chunk) <= 200


def test_hard_split_when_no_separator_exists():
    """一長串沒有任何分隔符的字元,最後手段是硬切。"""
    chunks = split_text("A" * 1000, chunk_size=300, overlap=0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]
    assert "".join(chunks) == "A" * 1000


def test_overlap_carries_tail_of_previous_chunk():
    text = "".join(f"第{i}句話。" for i in range(1, 100))
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        shared = max(
            (n for n in range(1, 51) if previous.endswith(current[:n])), default=0
        )
        assert shared > 0, "相鄰區塊之間應該有重疊"


def test_zero_overlap_produces_no_repetition():
    text = "".join(f"第{i}句話。" for i in range(1, 60))
    chunks = split_text(text, chunk_size=150, overlap=0)
    assert "".join(chunks) == text.replace("\n", "")


def test_paragraph_boundary_preferred_over_mid_sentence():
    a, b = "甲" * 100, "乙" * 100
    chunks = split_text(f"{a}\n\n{b}", chunk_size=120, overlap=0)
    assert len(chunks) == 2
    assert set(chunks[0]) == {"甲"} and set(chunks[1]) == {"乙"}


def test_separator_is_kept_with_preceding_text():
    """句號不該在切塊時被吃掉。"""
    text = "第一句。" * 80
    assert all(c.endswith("。") for c in split_text(text, chunk_size=100, overlap=0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"overlap": -1},
        {"overlap": DEFAULT_CHUNK_SIZE},
        {"chunk_size": 100, "overlap": 100},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        split_text("內容", **kwargs)
