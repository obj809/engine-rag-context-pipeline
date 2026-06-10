# Run-on-demand container for the REPL and the retrieval eval:
#   docker compose run --rm engine                            # REPL
#   docker compose run --rm engine python eval/run_eval.py    # eval (no OpenAI key)
#
# This is an ops tool, not a service: nothing listens, and the backend does NOT
# depend on this image (it copies the engine leaves into its own image at build
# time). The only runtime peer is Postgres on the vector-db compose network.
FROM python:3.12-slim

# Install torch from the CPU-only index first; PyPI's default wheels (amd64 AND
# aarch64) bundle the multi-GB CUDA/nvidia libraries, which this never uses.
# Keep this line identical to the backend Dockerfile's so the layer is shared.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bake the embedding model into the image (~130 MB) so container start needs no
# HuggingFace download. Must match EMBEDDING_MODEL in the indexing repo's
# build_index.py — rebuild with --build-arg if that changes.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"

COPY . /app/engine-rag-context-pipeline/

WORKDIR /app/engine-rag-context-pipeline
CMD ["python", "ask.py"]
