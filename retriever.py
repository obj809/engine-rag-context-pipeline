"""A LangChain retriever over the existing pgvector `chunks` table.

Wraps the same SQL cosine search as before in a BaseRetriever so it composes
into LCEL chains — without adopting LangChain's PGVector (which owns its own
schema). Embedding stays raw sentence-transformers, reusing the model the
orchestrator already loaded.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# BGE v1.5 asymmetric retrieval: queries get an instruction prefix, documents don't.
# If you swap to a model that doesn't use prefixes, set this to "".
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class PgVectorRetriever(BaseRetriever):
    conn: Any        # psycopg connection (register_vector already applied by the caller)
    embedder: Any    # sentence_transformers.SentenceTransformer
    k: int = 6

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        q = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        # Vectors are unit-normalized, so cosine distance (<=>) ranks identically to
        # cosine similarity — ascending distance == most similar first.
        rows = self.conn.execute(
            "SELECT content, page, volume FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (q, self.k),
        ).fetchall()
        return [
            Document(page_content=r[0], metadata={"page": r[1], "volume": r[2]})
            for r in rows
        ]
