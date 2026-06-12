"""Retrieval evaluation harness for the RAG pipeline.

Runs each ground-truth question through the same PgVectorRetriever the app uses
and reports retrieval quality — hit-rate@k and MRR — using the `page` each chunk
came from as the relevance label. No OpenAI key required: this measures retrieval
only (the part that decides whether the right context ever reaches the LLM), so it
runs offline and for free. Answer-quality scoring is a deliberate next increment.

Run:  python eval/run_eval.py                  (from this repo root; after building the index)
      python eval/run_eval.py --k 10 --show-misses
Env:  DATABASE_URL  (this repo's .env first, then the umbrella .env)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent   # engine-rag-context-pipeline/
UMBRELLA = REPO_ROOT.parent                            # rag-context-pipeline/ (transitional fallback)
# Reuse the exact retriever the app uses, so the eval reflects real behavior. The
# engine leaves are flattened at the repo root, so put that on sys.path.
sys.path.insert(0, str(REPO_ROOT))
from load_index import load_index          # noqa: E402
from retriever import PgVectorRetriever     # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
HIT_KS = (1, 3, 5, 6, 10)


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def first_hit_rank(retrieved_pages: list[int], expected: list[int]) -> int | None:
    """1-based rank of the first retrieved page that is a relevant page, else None."""
    expected_set = set(expected)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_set:
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10, help="chunks to retrieve per query (default 10)")
    ap.add_argument("--show-misses", action="store_true", help="print questions not hit within top-6")
    args = ap.parse_args()

    # Precedence: real environment > repo-local .env > umbrella .env — an
    # explicitly set env var is never overridden by a .env file.
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(UMBRELLA / ".env")
    dataset = load_dataset(DATASET)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        embedding_model = load_index(conn)
        print(f"Loading embedding model {embedding_model}...")
        model = SentenceTransformer(embedding_model)
        retriever = PgVectorRetriever(conn=conn, embedder=model, k=args.k)

        print(f"\nEvaluating {len(dataset)} questions (retrieving top-{args.k})\n")
        print(f"{'rank':>4}  {'hit@6':>5}  question")
        print("-" * 72)

        ranks: list[int | None] = []
        for item in dataset:
            docs = retriever.invoke(item["question"])
            pages = [d.metadata["page"] for d in docs]
            rank = first_hit_rank(pages, item["expected_pages"])
            ranks.append(rank)
            hit6 = "yes" if rank and rank <= 6 else "no"
            print(f"{(rank or '—'):>4}  {hit6:>5}  {item['question'][:56]}")
            if args.show_misses and (rank is None or rank > 6):
                print(f"        expected {item['expected_pages']}, got {pages[:6]}")

        n = len(ranks)
        print("-" * 72)
        for k in (k for k in HIT_KS if k <= args.k):
            hit_rate = sum(1 for r in ranks if r and r <= k) / n
            print(f"hit-rate@{k:<2}  {hit_rate:6.1%}")
        mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / n
        print(f"MRR        {mrr:6.3f}")


if __name__ == "__main__":
    main()
