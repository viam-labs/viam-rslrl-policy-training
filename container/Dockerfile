# Training image. For now it holds the sim package and the contact spike; the training plan adds
# mjlab and the training script on top. linux/amd64 is what Viam training jobs run.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY rl_insert rl_insert
COPY spikes spikes

CMD ["python", "spikes/contact_warp.py", "--device", "cuda"]
