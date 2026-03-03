# Optional Docker build for ARM64 (e.g. CI, portability)
# Note: MPS passthrough on Mac Docker is experimental; bare-metal preferred for M3 perf.

FROM --platform=linux/arm64 mambaorg/micromamba:1.5.1
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -afy

WORKDIR /app
COPY scripts/ scripts/
COPY configs/ configs/

CMD ["python", "scripts/distill_minillm.py", "--help"]
