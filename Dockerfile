# Stage 1: Build the binary and index tools
FROM rust:1.85-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY core ./core
COPY api ./api
COPY testsuite ./testsuite
RUN cargo build --release --bin rinha-api && \
    cargo build --release --bin build-index && \
    strip target/release/rinha-api target/release/build-index

# Stage 2: Build the pre-computed index from official dataset
FROM alpine:3.20 AS index-builder
RUN apk add --no-cache ca-certificates curl libgcc
WORKDIR /app
COPY --from=builder /app/target/release/build-index /app/build-index
RUN mkdir -p /app/data && \
    curl -sL https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz \
    -o /app/data/references.json.gz && \
    /app/build-index /app/data /app/data/index.bin.gz && \
    rm -f /app/build-index /app/data/references.json.gz

# Stage 3: Runtime — just the binary + pre-built index
FROM alpine:3.20
RUN apk add --no-cache ca-certificates
WORKDIR /app

COPY --from=builder /app/target/release/rinha-api /app/rinha-api
COPY --from=index-builder /app/data/index.bin.gz /app/data/index.bin.gz

EXPOSE 8080

ENV DATA_DIR=/app/data
ENV INDEX_PATH=/app/data/index.bin.gz
ENV PORT=8080

CMD ["/app/rinha-api"]
