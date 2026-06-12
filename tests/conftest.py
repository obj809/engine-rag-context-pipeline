"""Shared test fixtures: fakes for the engine's two boundaries (DB + LLM).

The leaves take their collaborators as arguments (the psycopg connection, the
embedder, the chat model), so the suite swaps in stand-ins and runs offline:
no database, no OpenAI key, no model download.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Make the repo-root leaves (retriever/chain/load_index) importable regardless
# of how pytest was invoked — pytest only puts this tests/ dir on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever import PgVectorRetriever

FAKE_ANSWER = "The target year is 2050 [page 96]."
FAKE_ROWS = [("Net zero by 2050 is the national commitment.", 96), ("Sector pathways differ.", 4)]


class FakeEmbedder:
    """Records every encode() call so tests can assert on prefix and kwargs."""

    def __init__(self):
        self.encoded: list[str] = []
        self.kwargs: list[dict] = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        self.kwargs.append(kwargs)
        return [0.0, 0.0, 0.0]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    """Records every execute() call and serves canned rows."""

    def __init__(self, rows):
        self._rows = rows
        self.queries: list[tuple] = []  # (sql, params)

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        return _FakeResult(self._rows)


class RecordingChatModel(BaseChatModel):
    """Answers FAKE_ANSWER and records the prompt messages it was invoked with."""

    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=FAKE_ANSWER))])


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def conn():
    return FakeConn(FAKE_ROWS)


@pytest.fixture
def retriever(conn, embedder):
    return PgVectorRetriever(conn=conn, embedder=embedder, k=2)
