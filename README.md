# Rinha de Backend 2026 — Rust

Fraud detection API using vector search with IVF (Inverted File Index).

- **Language:** Rust
- **Vector search:** IVF with 512 clusters, f16 SIMD (F16C + FMA)
- **JSON parsing:** Custom zero-alloc parser
- **Load balancer:** nginx (round-robin)
- **Allocator:** MiMalloc

## Structure

```
├── api/           # HTTP API (axum)
├── core/          # Vector search engine + dataset loading
├── testsuite/     # Accuracy tests
├── benches/       # Bench scripts
├── data/          # Dataset files (gitignored)
└── nginx/         # nginx config
```

## License

MIT
