# Optional Docker build for ARM64 (e.g. CI, portability)
# Note: MPS passthrough on Mac Docker is experimental; bare-metal preferred for M3 perf.

FROM --platform=linux/arm64 mambaorg/micromamba:1.5.1
ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -afy

WORKDIR /app
COPY distill/ distill/
COPY pyproject.toml pyproject.toml
COPY configs/ configs/
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "distill.distill_minillm", "--help"]
