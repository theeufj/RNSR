# rnsr service image.
#
# Two things this image exists to guarantee, beyond "it runs somewhere":
#
#  1. A non-root user. The sandbox audit hook (rnsr/env/fsguard.py) confines
#     model-written code to the corpus, but defence in depth means the process
#     it escapes into should own nothing worth taking. Run one container per
#     tenant and the filesystem boundary is the kernel's, not Python's.
#  2. No provider keys baked in. Keys arrive at runtime (env or secret mount)
#     and are dropped from the sandbox child's environment on spawn.
#
# Build:  docker build -t rnsr:latest .
# Serve:  docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=... \
#           -v "$PWD/runs:/data/runs" rnsr:latest
# CLI:    docker run --rm -e ANTHROPIC_API_KEY=... -v "$PWD:/work" \
#           rnsr:latest rnsr answer-csv --corpus /work/matter ...

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RNSR_LOG_FORMAT=json \
    RNSR_RUN_DIR=/data/runs

# Runtime deps only: the text-tier parsers (pdfium) need no system libraries,
# and the layout-ML tier is deliberately out of this image — a service that
# answers questions should not carry an ML ingest stack it may never use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY rnsr ./rnsr
RUN pip install --no-cache-dir ".[service,secure]"

# Least privilege: the process owns its data directory and nothing else.
RUN useradd --create-home --uid 10001 rnsr \
    && mkdir -p /data/runs \
    && chown -R rnsr:rnsr /data
USER rnsr

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; \
u.urlopen('http://127.0.0.1:8000/healthz').read()"

ENTRYPOINT ["tini", "--"]
CMD ["rnsr", "serve", "--host", "0.0.0.0", "--port", "8000"]
