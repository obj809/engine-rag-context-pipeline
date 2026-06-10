# engine-rag-context-pipeline

The query engine for the [RAG Context Pipeline](../): embeds a question, retrieves
the top-k chunks from the Postgres `chunks` table (pgvector cosine distance), and
composes the answer as a LangChain LCEL chain (`retriever | prompt | llm | parser`)
with inline `[page N]` citations. Ships with an interactive REPL and the retrieval
eval harness.

One of the per-concern repos the pipeline is split into (`backend-`, `engine-`,
`indexing-`, `vector-db-rag-context-pipeline`). This repo owns the **query side**.

## Contents

```
retriever.py      # PgVectorRetriever(BaseRetriever) — pgvector SQL cosine search + BGE query prefix
chain.py          # build_chain: retriever → prompt → LLM → text (LCEL)
load_index.py     # reads the embedding-model name recorded on the chunks table
ask.py            # interactive REPL orchestrator (opens the connection, builds + invokes the chain)
eval/
├── run_eval.py   # retrieval eval: hit-rate@k / MRR (no OpenAI key needed)
└── dataset.jsonl # ground-truth questions → expected pages
```

The retriever wraps raw pgvector SQL in a LangChain `BaseRetriever` rather than
using LangChain's `PGVector` vectorstore, so the `chunks` table schema (defined by
`indexing-rag-context-pipeline`) stays under the project's control. Embedding stays
raw `sentence-transformers`. **BGE asymmetric retrieval:** queries get the
`QUERY_PREFIX` instruction string in `retriever.py`; documents don't. If you swap
to a model that doesn't use prefixes, set `QUERY_PREFIX = ""`.

Both the backend API and this REPL build the same retriever + chain — the backend
imports these leaf modules over `sys.path` (until the engine is published as a
package).

## Run

Prerequisites: Postgres up (`cd ../vector-db-rag-context-pipeline && docker compose
up -d`) and the index built (`cd ../indexing-rag-context-pipeline && python
build_index.py`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in OPENAI_API_KEY (DATABASE_URL is preset)

python ask.py               # interactive REPL
python eval/run_eval.py     # retrieval eval (hit-rate@k / MRR; no OpenAI key needed)
python eval/run_eval.py --k 10 --show-misses
```

`ask.py` / `eval/run_eval.py` read this repo's `.env` first, falling back to the
umbrella `.env`, so an existing umbrella `.env` keeps them running with no extra setup.

## Tuning

| Constant | File | Default |
|---|---|---|
| `TOP_K` | `ask.py` | 6 chunks |
| `OPENAI_MODEL` | `ask.py` | `gpt-5.4-nano` |
| `QUERY_PREFIX` | `retriever.py` | BGE instruction prefix |

## Required environment variables

- `OPENAI_API_KEY` — answer generation (the REPL). The eval harness needs none.
- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching
  `vector-db-rag-context-pipeline/docker-compose.yml`.

## Note on eval

`eval/run_eval.py` replays `dataset.jsonl` through the same `PgVectorRetriever` the
app uses, scoring retrieval only (hit-rate@k, MRR) with each chunk's `page` as the
relevance label — no OpenAI key, runs offline. Questions whose answers live in
chart/figure pixels (not extracted by the indexer) are flagged
`"chart_dependent": true` in the dataset.
