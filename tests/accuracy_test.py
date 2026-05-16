#!/usr/bin/env python3
"""Test KNN accuracy: IVF with NPROBE vs exact brute-force KNN.

Loads references.json.gz, builds an in-memory index matching our Rust IVF,
then tests accuracy by running KNN on a subset of queries.

Usage: python3 tests/accuracy_test.py [--sample N] [--nprobe N]
"""
import argparse
import json
import gzip
import struct
import sys
import time
import math
import numpy as np
from collections import Counter

VECTOR_SCALE = 0.0001

def load_refs(path):
    """Load reference dataset."""
    print(f"Loading {path}...")
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    X = []
    y = []
    for entry in data:
        X.append(entry['vector'])
        y.append(1 if entry['label'] == 'fraud' else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.uint8)
    print(f"  {len(X)} vectors, {y.sum()} fraud / {len(y)-y.sum()} legit")
    return X, y


def load_index(path):
    """Load our IVF index and reconstruct cluster structure."""
    print(f"Loading index {path}...")
    with gzip.open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'IVF1', f"bad magic: {magic}"
        n = struct.unpack('<I', f.read(4))[0]
        k = struct.unpack('<I', f.read(4))[0]
        d = struct.unpack('<I', f.read(4))[0]
        print(f"  n={n} k={k} d={d}")

        # Read centroids SOA (d * k of f32)
        centroids_raw = f.read(d * k * 4)
        centroids_soa = np.frombuffer(centroids_raw, dtype=np.float32).copy()
        # Convert SOA back to AOS for easier comparison
        centroids = np.zeros((k, d), dtype=np.float32)
        for dim in range(d):
            centroids[:, dim] = centroids_soa[dim * k:(dim + 1) * k]
        
        # Offsets [k+1]
        offsets_raw = f.read((k + 1) * 4)
        offsets = np.frombuffer(offsets_raw, dtype=np.uint32).copy()
        
        # Labels (padded_n bytes)
        padded_n = offsets[k]
        labels = np.frombuffer(f.read(padded_n), dtype=np.uint8).copy()
        
        # Blocks
        blocks_raw = f.read((padded_n // 8) * d * 8 * 2)
        blocks = np.frombuffer(blocks_raw, dtype=np.int16).copy()
        blocks = blocks.reshape(-1, d, 8)
    
    return centroids, offsets, labels, blocks, n, k


def exact_knn(query, X, y, k=5):
    """Brute-force exact KNN on the full dataset."""
    diffs = X - query.reshape(1, -1)
    dists = np.sum(diffs * diffs, axis=1)
    nearest = np.argpartition(dists, k)[:k]
    nearest = nearest[np.argsort(dists[nearest])]
    return y[nearest], dists[nearest]


def ivf_knn(query, centroids, offsets, labels, blocks, nprobe=15, k=5):
    """IVF KNN matching our Rust implementation."""
    d = 14
    
    # 1. Compute centroid distances
    c_dists = np.sum((centroids - query.reshape(1, -1)) ** 2, axis=1)
    best_centroids = np.argpartition(c_dists, nprobe)[:nprobe]
    
    # 2. Scan blocks in selected centroids
    best_d = np.full(k, np.inf, dtype=np.float32)
    best_labels = np.zeros(k, dtype=np.uint8)
    
    for ci in best_centroids:
        start = offsets[ci]
        end = offsets[ci + 1]
        
        block_start = start // 8
        block_end = (end + 7) // 8
        
        for bi in range(block_start, block_end):
            block = blocks[bi]  # [14, 8]
            
            # Compute 8 distances
            for v in range(8):
                global_idx = bi * 8 + v
                if global_idx >= end:
                    break
                
                diff = block[:, v].astype(np.float32) * VECTOR_SCALE - query
                d2 = np.sum(diff * diff)
                
                if d2 < best_d[-1]:
                    best_d[-1] = d2
                    best_labels[-1] = labels[global_idx]
                    # Insertion sort
                    j = k - 1
                    while j > 0 and best_d[j] < best_d[j - 1]:
                        best_d[j], best_d[j - 1] = best_d[j - 1], best_d[j]
                        best_labels[j], best_labels[j - 1] = best_labels[j - 1], best_labels[j]
                        j -= 1
    
    return best_labels, best_d


def main():
    parser = argparse.ArgumentParser(description='Test IVF KNN accuracy')
    parser.add_argument('--sample', type=int, default=500, help='Number of test queries')
    parser.add_argument('--nprobe', type=int, default=15, help='Number of centroid probes')
    parser.add_argument('--refs', default='data/references.json.gz', help='Reference dataset path')
    parser.add_argument('--index', default='data/index.bin.gz', help='Index file path')
    args = parser.parse_args()

    X_all, y_all = load_refs(args.refs)
    centroids, offsets, labels, blocks, n, k = load_index(args.index)
    
    # Sample queries: take evenly spaced indices
    total = len(X_all)
    step = max(1, total // args.sample)
    indices = list(range(0, total, step))[:args.sample]
    queries = X_all[indices]
    true_labels = y_all[indices]
    
    print(f"\nTesting {len(queries)} queries with NPROBE={args.nprobe}...")
    
    errors = 0
    fp = 0  # False positives: model says fraud (>=3/5), should be legit
    fn = 0  # False negatives: model says legit (<3/5), should be fraud
    n_fraud_queries = 0
    total_time = 0.0
    
    for qi, (q_idx, q, true_y) in enumerate(zip(indices, queries, true_labels)):
        if true_y == 1:
            n_fraud_queries += 1
        
        # Exact KNN
        t0 = time.perf_counter()
        exact_labels, exact_dists = exact_knn(q, X_all, y_all)
        exact_time = time.perf_counter() - t0
        
        # IVF KNN
        t0 = time.perf_counter()
        ivf_labels, ivf_dists = ivf_knn(q, centroids, offsets, labels, blocks, args.nprobe)
        ivf_time = time.perf_counter() - t0
        
        total_time += ivf_time
        
        # Compare results
        exact_fraud = (exact_labels.sum() >= 3)  # 3+ fraud out of 5 = approved fraud
        ivf_fraud = (ivf_labels.sum() >= 3)
        
        # Also compare the 5-th neighbor distance (best_d[4])
        exact_best5_dist = exact_dists[4]
        ivf_best5_dist = ivf_dists[4]
        found_exact5 = any(ivf_labels[i] == exact_labels[i] for i in range(5))
        
        if exact_fraud != ivf_fraud:
            errors += 1
            if ivf_fraud and not exact_fraud:
                fp += 1
            else:
                fn += 1
            
            if qi < 10:  # Print first 10 mismatches
                print(f"  MISMATCH q[{qi}] (idx={q_idx}), true={bool(true_y)}")
                print(f"    Exact:  labels={exact_labels}, dists={[f'{d:.6f}' for d in exact_dists]}")
                print(f"    IVF:    labels={ivf_labels}, dists={[f'{d:.6f}' for d in ivf_dists]}")
        
        if qi > 0 and qi % 100 == 0:
            print(f"  Progress: {qi}/{len(queries)}, errors={errors}, fp={fp}, fn={fn}")
    
    total_queries = len(queries)
    accuracy = (total_queries - errors) / total_queries * 100
    fp_rate = fp / max(total_queries, 1) * 100
    fn_rate = fn / max(n_fraud_queries, 1) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS: NPROBE={args.nprobe}")
    print(f"{'='*60}")
    print(f"  Queries tested:     {total_queries}")
    print(f"  Fraud queries:      {n_fraud_queries}")
    print(f"  Mismatches:         {errors} ({100-accuracy:.2f}%)")
    print(f"  Accuracy:           {accuracy:.2f}%")
    print(f"  False Positives:    {fp} ({fp_rate:.2f}%)")
    print(f"  False Negatives:    {fn} ({fn_rate:.2f}%)")
    print(f"  Avg IVF time:       {total_time/total_queries*1000:.2f}ms")
    print(f"  Avg exact time:     0.0ms (pre-computed)" if False else "")
    
    if errors > 0:
        print(f"\n⚠️  WARNING: NPROBE={args.nprobe} causes {errors} mismatches!")
        print(f"   {'↑ Might FP/FN in competition!' if fp+fn > 0 else ''}")
    else:
        print(f"\n✅ PERFECT: NPROBE={args.nprobe} matches exact KNN on all sampled queries!")
    
    # Additional detail
    if errors > 0:
        fp_in_detection = sum(1 for idx in indices if 
            y_all[idx] == 0 and  # true legit
            sum(ivf_knn(X_all[idx], centroids, offsets, labels, blocks, args.nprobe)[0]) >= 3  # model says fraud
        )
        fn_in_detection = sum(1 for idx in indices if 
            y_all[idx] == 1 and  # true fraud
            sum(ivf_knn(X_all[idx], centroids, offsets, labels, blocks, args.nprobe)[0]) < 3  # model says legit
        )
        print(f"\n  Detection-level FP: {fp_in_detection}")
        print(f"  Detection-level FN: {fn_in_detection}")


if __name__ == '__main__':
    main()
