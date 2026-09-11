import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from chat_history import get_user_history, get_last_n_messages, embed_messages, DB_PATH


@pytest.fixture
def seeded_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS history (user_id TEXT, message TEXT, response TEXT)")
    conn.execute("DELETE FROM history")
    conn.executemany(
        "INSERT INTO history (user_id, message, response) VALUES (?, ?, ?)",
        [("u1", "hi", "hello"), ("u1", "how are you", "good"), ("u1", "bye", "goodbye")],
    )
    conn.commit()
    conn.close()
    yield


def test_get_user_history_returns_rows(seeded_db):
    assert len(get_user_history("u1")) == 3


def test_get_user_history_requires_user_id():
    with pytest.raises(ValueError):
        get_user_history("")


def test_get_last_n_messages_empty_history(seeded_db):
    assert get_last_n_messages("nonexistent_user", 5) == []


def test_get_last_n_messages_n_greater_than_history(seeded_db):
    result = get_last_n_messages("u1", 100)
    assert len(result) == 3  # doesn't crash, just returns what exists


def test_get_last_n_messages_zero_or_negative(seeded_db):
    assert get_last_n_messages("u1", 0) == []
    assert get_last_n_messages("u1", -1) == []


@patch("chat_history.requests.post")
def test_embed_messages_batches_single_call(mock_post):
    mock_post.return_value = MagicMock(
        json=lambda: {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
    )
    mock_post.return_value.raise_for_status = lambda: None

    result = embed_messages(["hello", "world"])

    assert mock_post.call_count == 1  # one batched call, not N calls
    assert len(result) == 2


def test_embed_messages_empty_input():
    assert embed_messages([]) == []