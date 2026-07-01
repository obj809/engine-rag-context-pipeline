"""Orchestrator: interactive RAG Q&A against the pgvector `chunks` table.

Run:  python ask.py     (from this repo root)
Deps: langchain-core, langchain-openai, sentence-transformers, numpy, psycopg, pgvector
Env:  OPENAI_API_KEY, DATABASE_URL  (this repo's .env first, then the umbrella .env)
      OPENAI_BASE_URL (optional) — point at a LiteLLM/OpenAI-compatible proxy;
      when set, OPENAI_API_KEY is that proxy's key. Unset = direct OpenAI.
Note: build the index first — cd ../indexing-rag-context-pipeline && python build_index.py
      (and start the DB — cd ../vector-db-rag-context-pipeline && docker compose up -d).
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from chain import build_chain
from load_index import load_index
from retriever import PgVectorRetriever

REPO_ROOT = Path(__file__).resolve().parent
UMBRELLA = REPO_ROOT.parent
TOP_K = 6
OPENAI_MODEL = "gpt-5.4-mini"


def main() -> None:
    # Precedence: real environment > repo-local .env > umbrella .env — an
    # explicitly set env var is never overridden by a .env file.
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(UMBRELLA / ".env")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        embedding_model = load_index(conn)
        print(f"Loading embedding model {embedding_model}...")
        model = SentenceTransformer(embedding_model)

        retriever = PgVectorRetriever(conn=conn, embedder=model, k=TOP_K)
        # base_url unset → direct OpenAI; set → route via a LiteLLM/OpenAI-compatible proxy.
        llm = ChatOpenAI(model=OPENAI_MODEL, base_url=os.getenv("OPENAI_BASE_URL") or None)
        chain = build_chain(retriever, llm)

        print("Ready. Ask a question (Ctrl-C / empty line to exit).")
        while True:
            try:
                query = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not query:
                return
            print()
            print(chain.invoke(query))


if __name__ == "__main__":
    main()
