"""Read the embedding-model name recorded alongside the index in Postgres.

The query side must embed with the same model used at index time; that name is
stored on the chunks table (co-located with the data) rather than hardcoded here.
"""

import psycopg


def load_index(conn: psycopg.Connection) -> str:
    row = conn.execute("SELECT embedding_model FROM chunks LIMIT 1").fetchone()
    if row is None:
        raise SystemExit(
            "No chunks in the database — run: "
            "cd indexing-rag-context-pipeline && python build_index.py"
        )
    return row[0]
