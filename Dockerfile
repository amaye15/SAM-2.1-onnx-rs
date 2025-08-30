# syntax=docker/dockerfile:1.6

# ---------- Build stage ----------
FROM rust:1-bookworm AS builder
WORKDIR /app

# Speed up builds and improve caching:
# 1) Copy only manifests first
COPY Cargo.toml Cargo.lock ./
COPY sam2_server/Cargo.toml sam2_server/Cargo.toml

# 2) Create a dummy main to cache dependency compilation
RUN mkdir -p sam2_server/src \
 && printf "fn main(){}\n" > sam2_server/src/main.rs

# 3) Pre-build dependencies using BuildKit cache mounts (ignore failure on missing sources)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build -p sam2_server --release --locked || true

# 4) Now copy the actual sources for the server crate only (avoids invalidating cache on unrelated files)
COPY sam2_server/ ./sam2_server/

# 5) Build the release binary with caching
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build -p sam2_server --release --locked

# ---------- Runtime stage ----------
FROM debian:bookworm-slim AS runtime
WORKDIR /app

# Small, useful tools and certs (for HEALTHCHECK)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Copy binary and static assets
COPY --from=builder /app/target/release/sam2_server /usr/local/bin/sam2_server
COPY sam2_server/static ./sam2_server/static

# Copy ONNX models to workdir root (the server expects them next to /app)
COPY sam2_large.onnx sam2_small.onnx sam2_base_plus.onnx sam2_tiny.onnx ./

# Run as non-root
RUN useradd -r -u 10001 -g root -d /nonexistent -s /usr/sbin/nologin appuser \
 && chown -R appuser:root /app \
 && chmod -R g=u /app

ENV RUST_LOG=info \
    RUST_BACKTRACE=1

EXPOSE 8080

# Basic liveness probe against the models endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/api/models || exit 1

USER appuser
ENTRYPOINT ["/usr/local/bin/sam2_server"]
