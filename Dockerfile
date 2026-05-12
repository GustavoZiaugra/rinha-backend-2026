# Stage 1: Build
FROM rust:1.85-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml ./
COPY api ./api
COPY lb ./lb
COPY data/index.bin.gz ./data/index.bin.gz
RUN rustup target add x86_64-unknown-linux-musl
ENV RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2,+fma,+f16c,+bmi2,+popcnt"
RUN cargo build --release --target x86_64-unknown-linux-musl -p api -p lb
RUN strip target/x86_64-unknown-linux-musl/release/api
RUN strip target/x86_64-unknown-linux-musl/release/lb

# Stage 2: Runtime
FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/api /api
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/lb /lb
COPY --from=builder /app/data/index.bin.gz /data/index.bin.gz
ENV BIND=0.0.0.0:8080
EXPOSE 8080
CMD ["/api"]
