#!/usr/bin/env python3
"""Build IVF index for Rinha 2026 using numpy."""
import gzip
import json
import struct
import sys
import numpy as np

DIMS = 14
K = 1024
MAX_ITERS = 15
SCALE = 10000.0
SAMPLE_SIZE = 10000


def kmeans_plus_plus(X, k, rng_seed=0xDEADBEEF):
    n, d = X.shape
    rng = np.random.default_rng(rng_seed)
    centroids = np.empty((k, d), dtype=np.float32)
    centroids[0] = X[rng.integers(n)]
    dists = np.full(n, np.inf, dtype=np.float32)

    for i in range(1, k):
        d = np.sum((X - centroids[i - 1]) ** 2, axis=1)
        np.minimum(dists, d, out=dists)
        probs = dists / dists.sum()
        centroids[i] = X[rng.choice(n, p=probs)]
    return centroids


def lloyd(X, centroids, max_iters=25):
    n = X.shape[0]
    k = centroids.shape[0]
    assignments = np.empty(n, dtype=np.int32)
    chunk_size = 256
    for it in range(max_iters):
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dists = np.sum((X[start:end, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments[start:end] = np.argmin(dists, axis=1)
        for c in range(k):
            mask = assignments == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)
        print(f"  iter {it+1} done", file=sys.stderr)
    return centroids, assignments


def assign_all(X, centroids):
    n = X.shape[0]
    k = centroids.shape[0]
    assignments = np.empty(n, dtype=np.int32)
    chunk_size = 256
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = np.sum((X[start:end, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments[start:end] = np.argmin(dists, axis=1)
    return assignments


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{data_dir}/index.bin.gz"

    print("Loading references...", file=sys.stderr)
    with gzip.open(f"{data_dir}/references.json.gz", "rt") as f:
        entries = json.load(f)

    vectors = np.array([entry["vector"] for entry in entries], dtype=np.float32)
    labels = np.array([1 if entry["label"] == "fraud" else 0 for entry in entries], dtype=np.uint8)
    n = vectors.shape[0]
    print(f"  {n} vectors loaded", file=sys.stderr)

    step = max(1, n // SAMPLE_SIZE)
    sample = vectors[::step][:SAMPLE_SIZE].copy()
    print(f"K-means++ on {len(sample)} sample...", file=sys.stderr)
    centroids = kmeans_plus_plus(sample, K)
    print("Lloyd on sample...", file=sys.stderr)
    centroids, _ = lloyd(sample, centroids, MAX_ITERS)

    print("Assigning all vectors...", file=sys.stderr)
    assignments = assign_all(vectors, centroids)

    cluster_lists = [[] for _ in range(K)]
    for i, a in enumerate(assignments):
        cluster_lists[a].append(i)

    offsets = [0] * (K + 1)
    padded_labels = bytearray()
    all_blocks = []

    for c in range(K):
        offsets[c] = len(padded_labels)
        members = cluster_lists[c]
        clabels = bytearray(labels[i] for i in members)
        pad = (8 - len(clabels) % 8) % 8
        clabels.extend(b'\x00' * pad)
        padded_labels.extend(clabels)

        padded_count = len(clabels)
        num_blocks = padded_count // 8
        if num_blocks == 0:
            continue
        vecs = vectors[members]
        if len(members) < padded_count:
            pad_vecs = np.zeros((pad, DIMS), dtype=np.float32)
            vecs = np.vstack([vecs, pad_vecs])
        qvecs = np.rint(vecs * SCALE).astype(np.int16)
        qvecs = qvecs.reshape(num_blocks, 8, DIMS).transpose(0, 2, 1)
        all_blocks.append(qvecs)

    offsets[K] = len(padded_labels)

    print("Writing index...", file=sys.stderr)
    with gzip.open(output_path, "wb") as f:
        f.write(b"IVF1")
        f.write(struct.pack("<III", n, K, DIMS))

        for d in range(DIMS):
            f.write(struct.pack(f"<{K}f", *centroids[:, d].tolist()))

        f.write(struct.pack(f"<{K+1}I", *offsets))
        f.write(padded_labels)

        if all_blocks:
            blocks_arr = np.concatenate(all_blocks, axis=0)
            f.write(blocks_arr.astype(np.int16).tobytes())

    print(f"Done: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
