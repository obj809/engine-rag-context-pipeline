"""Unit tests for PgVectorRetriever — the SQL/embedding contract the chain
(and the backend, which imports this module over the sys.path bridge) rely on."""

from conftest import FAKE_ROWS

from retriever import QUERY_PREFIX


def test_rows_become_documents_with_page_and_volume_metadata(retriever):
    docs = retriever.invoke("what is the target year?")
    assert [
        (d.page_content, d.metadata["page"], d.metadata["volume"]) for d in docs
    ] == FAKE_ROWS


def test_query_gets_bge_instruction_prefix(retriever, embedder):
    retriever.invoke("what is the target year?")
    assert embedder.encoded == [QUERY_PREFIX + "what is the target year?"]


def test_embedding_is_normalized(retriever, embedder):
    retriever.invoke("anything")
    assert embedder.kwargs[0] == {"normalize_embeddings": True}


def test_sql_orders_by_cosine_distance_with_limit_k(retriever, conn, embedder):
    retriever.invoke("anything")
    sql, params = conn.queries[0]
    assert "ORDER BY embedding <=>" in sql
    assert params == ([0.0, 0.0, 0.0], 2)  # (the encoded query vector, k)


def test_k_defaults_to_six(conn, embedder):
    from retriever import PgVectorRetriever

    PgVectorRetriever(conn=conn, embedder=embedder).invoke("anything")
    assert conn.queries[0][1][1] == 6
