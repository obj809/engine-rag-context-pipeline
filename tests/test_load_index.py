"""Unit tests for load_index — the embedding_model column contract."""

import pytest

from conftest import FakeConn
from load_index import load_index


def test_returns_the_recorded_model_name():
    conn = FakeConn([("BAAI/bge-small-en-v1.5", None)])
    assert load_index(conn) == "BAAI/bge-small-en-v1.5"


def test_empty_table_exits_with_rebuild_hint():
    with pytest.raises(SystemExit, match="build_index.py"):
        load_index(FakeConn([]))
