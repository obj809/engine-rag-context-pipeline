# Engine RAG Context Pipeline

[![tests](https://github.com/obj809/engine-rag-context-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/obj809/engine-rag-context-pipeline/actions/workflows/tests.yml)

The query engine for the [RAG Context Pipeline](../): embeds a question, retrieves
the top-k chunks from the Postgres `chunks` table (pgvector cosine distance), and
composes the answer as a LangChain LCEL chain (`retriever | prompt | llm | parser`)
with inline `[Volume N, p.M]` citations. Ships with an interactive REPL and the
retrieval eval harness.

One of the per-concern repos the pipeline is split into (`backend-`, `engine-`,
`indexing-`, `vector-db-rag-context-pipeline`). This repo owns the **query side**.

## Contents

```
retriever.py      # PgVectorRetriever(BaseRetriever) — pgvector SQL cosine search + BGE query prefix
chain.py          # build_chain: retriever → prompt → LLM → text (LCEL); also exports the halves (format_docs, build_answer_chain) for streaming consumers
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
imports these leaf modules from the installed `rag-engine` package (this repo,
packaged via `pyproject.toml` and pinned in the backend's `requirements.txt`). The
backend's streaming `/chat` endpoint uses the chain's exposed halves
(`format_docs` + `build_answer_chain`) so it can retrieve eagerly and stream just
the LLM part.

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

## Run in Docker

Same prerequisites as above (database container up, index built host-side). The
container is **run-on-demand** — nothing listens, so there's no `up -d`:

```bash
docker compose run --rm engine                            # REPL (needs OPENAI_API_KEY)
docker compose run --rm engine python eval/run_eval.py    # eval (no key needed)
docker compose run --rm engine python eval/run_eval.py --k 10 --show-misses
```

Notes on how the image works:

- This is an **ops tool, not part of the serving path**: the backend API does not
  depend on this image — it installs the engine as the pinned `rag-engine` package
  from its Git repo at build time. The engine container's only runtime peer is Postgres.
- It joins two **external** networks: the vector-db repo's Compose network
  (`vector-db-rag-context-pipeline_default`) to reach Postgres as `db:5432`, and
  the shared `webnet` bridge (so it sits on the same host-wide network as the
  project's other containers). `DATABASE_URL` is set in `docker-compose.yml`; both
  networks must already exist (`webnet` via `docker network create webnet`) before
  `docker compose run`.
- `OPENAI_API_KEY` is interpolated from **this repo's** `.env` — the umbrella-`.env`
  fallback doesn't apply inside the container, so the real key must live here
  (the eval needs no key at all).
- The embedding model (`BAAI/bge-small-en-v1.5`) is baked into the image at build
  time; the torch-install layer is written to match the backend's Dockerfile so the
  multi-GB layer is shared between the two images. If you change `EMBEDDING_MODEL`
  in the indexer, rebuild with `--build-arg EMBEDDING_MODEL=...`.
- `.env` files are deliberately excluded from the image
  (`Dockerfile.dockerignore`): a baked-in `.env` would override the injected
  `DATABASE_URL` and embed the API key.

## Tests

Unit tests for the three leaves (`tests/`, fakes in `tests/conftest.py`): the
retriever's SQL/prefix/normalization contract, `format_docs` page citations, the
chain halves against a fake LLM (including that `build_answer_chain` actually
streams — the backend's `/chat` depends on it), and `load_index`'s
`embedding_model` lookup. They run **offline**: no database, no OpenAI key, no
model download. (`ask.py` is interactive glue and isn't covered; retrieval
*quality* is the eval harness's job, below.)

```bash
pip install -r requirements-dev.txt   # pytest, on top of requirements.txt
python -m pytest
```

## Tuning

| Constant | File | Default |
|---|---|---|
| `TOP_K` | `ask.py` | 6 chunks |
| `OPENAI_MODEL` | `ask.py` | `gpt-5.4-mini` |
| `QUERY_PREFIX` | `retriever.py` | BGE instruction prefix |

## Required environment variables

- `OPENAI_API_KEY` — answer generation (the REPL). The eval harness needs none.
  With `OPENAI_BASE_URL` set this is the LiteLLM proxy's key, not the raw OpenAI key.
- `OPENAI_BASE_URL` — **optional**. Unset → the REPL calls OpenAI directly. Set
  (e.g. `http://localhost:4000/v1`) → routes through a LiteLLM/OpenAI-compatible proxy.
- `DATABASE_URL` — e.g. `postgresql://rag:rag@localhost:5432/rag`, matching
  `vector-db-rag-context-pipeline/docker-compose.yml`.

## Note on eval

`eval/run_eval.py` replays `dataset.jsonl` through the same `PgVectorRetriever` the
app uses, scoring retrieval only (hit-rate@k, MRR) — no OpenAI key, runs offline.
Pages restart per volume, so the relevance label is a `(volume, page)` pair: each
row's `expected` lists the `[volume, page]` location(s) where the answer text sits
in the Act.

`dataset.jsonl` holds 12 EPBC questions (MNES, the water trigger, penalties, the
referral process, the Australian Whale Sanctuary, conservation agreements, …).
Baseline on the current index: **hit-rate@1 50%, hit-rate@3/6/10 ≈ 92%, MRR ≈ 0.65**.
The one consistent miss is "objects of the Act" (s 3) — "objects of this Part"
recurs throughout the Act, so the s 3 page doesn't surface in the top-k.
