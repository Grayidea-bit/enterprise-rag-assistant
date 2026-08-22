"""api/conversations.py 的歷史重建。"""

from pydantic_ai.messages import ModelRequest, ModelResponse

from api.conversations import MAX_HISTORY_MESSAGES, to_history


def rows(n: int) -> list[dict]:
    """交錯的 user / assistant 回合。"""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"訊息{i}"} for i in range(n)
    ]


def test_empty_history():
    assert to_history([]) == []


def test_roles_map_to_the_right_message_types():
    history = to_history(rows(4))
    assert [type(m) for m in history] == [
        ModelRequest,
        ModelResponse,
        ModelRequest,
        ModelResponse,
    ]


def test_content_is_preserved_in_order():
    history = to_history(rows(4))
    assert [m.parts[0].content for m in history] == ["訊息0", "訊息1", "訊息2", "訊息3"]


def test_history_is_capped():
    """不設上限的話 context 會隨對話無限膨脹。"""
    history = to_history(rows(MAX_HISTORY_MESSAGES * 3))
    assert len(history) == MAX_HISTORY_MESSAGES


def test_cap_keeps_the_most_recent_messages():
    total = MAX_HISTORY_MESSAGES + 6
    history = to_history(rows(total))
    assert history[-1].parts[0].content == f"訊息{total - 1}"
    assert history[0].parts[0].content == f"訊息{total - MAX_HISTORY_MESSAGES}"


def test_sources_are_not_replayed_to_the_model():
    """來源只給前端顯示,不該塞回 context 佔 token。"""
    row = {"role": "assistant", "content": "答案", "sources": [{"source": "a.md"}]}
    (message,) = to_history([row])
    assert len(message.parts) == 1
    assert message.parts[0].content == "答案"
