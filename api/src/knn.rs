use crate::data::Dataset;
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

const MAX_CENTROIDS: usize = 4096;
const VECTOR_SCALE: f32 = 0.0001;
const KNN_K: usize = 5;

/// Proven SSE-based KNN with K=4096 and NPROBE=15.
pub fn knn5_fraud_count(query: &[f32; 14], ds: &Dataset) -> u8 {
    unsafe { knn5_ivf(query, ds) }
}

/// Runtime-variant NPROBE for accuracy testing.
/// Uses heap-allocated probes; never called on the hot path.
pub fn knn5_fraud_count_nprobe(query: &[f32; 14], ds: &Dataset, nprobe: usize) -> u8 {
    unsafe {
        let mut dists = [MaybeUninit::<f32>::uninit(); MAX_CENTROIDS];
        compute_centroid_dists(query, ds, &mut dists);

        let mut q_i16 = [0i16; 14];
        for d in 0..14usize {
            q_i16[d] = (query[d] / VECTOR_SCALE).round() as i16;
        }

        let probes = top_k_indices(&dists, ds.k, nprobe);
        let mut best_d = [f32::MAX; KNN_K];
        let mut best_id = [0u32; KNN_K];
        let mut best_label = [0u8; KNN_K];
        scan_raw(&probes, ds, &q_i16, &mut best_d, &mut best_id, &mut best_label);
        best_label.iter().filter(|&&l| l == 1).count() as u8
    }
}

unsafe fn top_k_indices(dists: &[MaybeUninit<f32>; MAX_CENTROIDS], k: usize, n: usize) -> Vec<usize> {
    let dp = dists.as_ptr() as *const f32;
    let mut best = vec![(f32::MAX, 0usize); n];
    for i in 0..k {
        let d = *dp.add(i);
        if d < best[n - 1].0 {
            best[n - 1] = (d, i);
            let mut j = n - 1;
            while j > 0 && best[j].0 < best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(best[i].1);
    }
    out
}

/// Warmup: page-touch centroids, offsets, blocks, and labels for TLB/cache residency.
/// Then runs 20 synthetic queries to prime I-cache and branch predictor.
/// No SIMD needed — no target_feature annotation required.
pub fn warmup() {
    let ds = crate::data::dataset();
    let mut sink: u64 = 0;

    // Page-touch centroids (224KB) and offsets (16KB) for TLB residency
    for v in ds.centroids.iter() {
        sink ^= v.to_bits() as u64;
    }
    for v in ds.offsets.iter() {
        sink ^= *v as u64;
    }

    // Page-touch labels (~300KB) and blocks (~9MB) to prevent cold page faults
    // on the first real request. Touching every 1024th entry is enough to
    // populate the TLB without scanning every byte.
    for (i, &l) in ds.labels.iter().enumerate().step_by(1024) {
        sink ^= l as u64;
        // Touch corresponding position in blocks too
        let block_idx = i / 8;
        if block_idx * 14 * 8 < ds.blocks.len() {
            sink ^= unsafe { *ds.blocks.as_ptr().add(block_idx * 14 * 8) as u64 };
        }
    }

    let _ = sink;

    // 20 quick synthetic queries for I-cache and branch predictor priming
    let mut state = 0x12345678u32;
    for _ in 0..20 {
        let mut q = [0.0f32; 14];
        for v in q.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (state >> 8) as f32 / (1u32 << 24) as f32;
        }
        let _ = knn5_fraud_count(&q, ds);
    }
}

/// Entry point: two-tier KNN with FAST=5, FULL=10.
/// Tagged with avx2+fma because compute_centroid_dists uses FMA.
#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_ivf(query: &[f32; 14], ds: &Dataset) -> u8 {
    let mut dists = [MaybeUninit::<f32>::uninit(); MAX_CENTROIDS];
    compute_centroid_dists(query, ds, &mut dists);

    // Quantize query to i16 for integer SSE scan
    let mut q_i16 = [0i16; 14];
    for d in 0..14usize {
        q_i16[d] = (query[d] / VECTOR_SCALE).round() as i16;
    }

    // Two-tier search: always compute top 10, but try with 5 first.
    // If clearly legit (0-1 fraud) or clearly fraud (4-5), return fast.
    // Only ambiguous cases (2-3 fraud) scan the remaining 5 centroids.
    const FAST_N: usize = 5;
    const FULL_N: usize = 10;

    let all_probes = top_n_from_dists::<FULL_N>(&dists, ds.k);

    // Shared best-N state across tiers
    let mut best_d = [f32::MAX; KNN_K];
    let mut best_id = [0u32; KNN_K];
    let mut best_label = [0u8; KNN_K];

    // Tier 1: scan closest 5 centroids (~365 vectors)
    scan_raw(
        &all_probes[..FAST_N],
        ds,
        &q_i16,
        &mut best_d,
        &mut best_id,
        &mut best_label,
    );

    let fast_count = best_label.iter().filter(|&&l| l == 1).count() as u8;
    if fast_count <= 1 || fast_count >= 4 {
        return fast_count;
    }

    // Tier 2: scan next 5 centroids, keeping Tier 1's best distances.
    // Only reached for ~20-30% of queries (ambiguous fraud likelihood).
    scan_raw(
        &all_probes[FAST_N..],
        ds,
        &q_i16,
        &mut best_d,
        &mut best_id,
        &mut best_label,
    );

    best_label.iter().filter(|&&l| l == 1).count() as u8
}

/// AVX2 + FMA centroid distance computation.
/// This is the ONLY function that genuinely needs FMA (_mm256_fmadd_ps).
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_centroid_dists(
    query: &[f32; 14],
    ds: &Dataset,
    dists: &mut [MaybeUninit<f32>; MAX_CENTROIDS],
) {
    let k = ds.k;
    let cp = ds.centroids.as_ptr();
    let dp = dists.as_mut_ptr() as *mut f32;

    {
        let qd = _mm256_set1_ps(query[0]);
        let mut ci = 0usize;
        while ci + 16 <= k {
            // Prefetch next centroids to overlap memory latency
            _mm_prefetch(cp.add(ci + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(cp.add(ci + 40) as *const i8, _MM_HINT_T0);
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci)), qd);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci + 8)), qd);
            let s0 = _mm256_mul_ps(d0, d0);
            let s1 = _mm256_mul_ps(d1, d1);
            _mm256_storeu_ps(dp.add(ci), s0);
            _mm256_storeu_ps(dp.add(ci + 8), s1);
            ci += 16;
        }
        while ci + 8 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci)), qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(ci) - query[0];
            *dp.add(ci) = diff * diff;
            ci += 1;
        }
    }

    for dim in 1..14usize {
        let qd = _mm256_set1_ps(query[dim]);
        let base = dim * k;
        let mut ci = 0usize;
        while ci + 16 <= k {
            // Prefetch next centroids + accumulated distances
            _mm_prefetch(cp.add(base + ci + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(cp.add(base + ci + 40) as *const i8, _MM_HINT_T0);
            _mm_prefetch(dp.add(ci + 32) as *const i8, _MM_HINT_T0);
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(base + ci)), qd);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(base + ci + 8)), qd);
            let s0 = _mm256_fmadd_ps(d0, d0, _mm256_loadu_ps(dp.add(ci)));
            let s1 = _mm256_fmadd_ps(d1, d1, _mm256_loadu_ps(dp.add(ci + 8)));
            _mm256_storeu_ps(dp.add(ci), s0);
            _mm256_storeu_ps(dp.add(ci + 8), s1);
            ci += 16;
        }
        while ci + 8 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(base + ci)), qd);
            let s0 = _mm256_fmadd_ps(d0, d0, _mm256_loadu_ps(dp.add(ci)));
            _mm256_storeu_ps(dp.add(ci), s0);
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(base + ci) - query[dim];
            *dp.add(ci) += diff * diff;
            ci += 1;
        }
    }
}

/// Simple top-N selection — no SIMD, no target_feature needed.
#[inline(always)]
unsafe fn top_n_from_dists<const N: usize>(
    dists: &[MaybeUninit<f32>; MAX_CENTROIDS],
    k: usize,
) -> [usize; N] {
    let dp = dists.as_ptr() as *const f32;
    let mut best = [(f32::MAX, 0usize); N];
    for i in 0..k {
        let d = *dp.add(i);
        if d < best[N - 1].0 {
            best[N - 1] = (d, i);
            let mut j = N - 1;
            while j > 0 && best[j].0 < best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    let mut out = [0usize; N];
    for i in 0..N {
        out[i] = best[i].1;
    }
    out
}

/// AVX2 block scan into existing best-N state.
/// Accumulates distances into `best_d/best_id/best_label` without re-initializing.
/// Used by two-tier KNN to carry state between tiers.
///
/// Uses integer AVX2 (cvtepi16_epi32, sub_epi32, mullo_epi32)
/// but NOT FMA — avoids any potential AVX-FMA frequency downclock.
#[target_feature(enable = "avx2")]
unsafe fn scan_raw(
    probes: &[usize],
    ds: &Dataset,
    q_i16: &[i16; 14],
    best_d: &mut [f32; KNN_K],
    best_id: &mut [u32; KNN_K],
    best_label: &mut [u8; KNN_K],
) {
    let blocks = ds.blocks.as_ptr();
    let labels = ds.labels.as_ptr();
    let block_stride = 14 * 8;

    for &ci in probes {
        let start = ds.offsets[ci] as usize;
        let end = ds.offsets[ci + 1] as usize;
        let block_start = start / 8;
        let block_end = (end + 7) / 8;

        let mut bi = block_start;
        while bi < block_end {
            // Prefetch next block while processing current one
            if bi + 1 < block_end {
                let next = blocks.add((bi + 1) * block_stride) as *const i8;
                _mm_prefetch(next, _MM_HINT_T0);
            }

            let block_ptr = blocks.add(bi * block_stride);
            // SIMD-accumulate squared distances in AVX2 register,
            // extract only once after all 14 dimensions — saves 13× extract/transmute/scalar-accumulate
            // per block. Overflow check: each dim max 10000^2 = 1e8; ×14 = 1.4e9 < i32 max 2.147e9 ✓
            let mut dists = _mm256_setzero_si256();
            for d in 0..14usize {
                let qv = _mm256_set1_epi32(q_i16[d] as i32);
                let vals = _mm_loadu_si128(block_ptr.add(d * 8) as *const __m128i);
                let ve = _mm256_cvtepi16_epi32(vals);
                let diff = _mm256_sub_epi32(ve, qv);
                let sq = _mm256_mullo_epi32(diff, diff);
                dists = _mm256_add_epi32(dists, sq);
            }

            // Single extraction at the end
            let dists_lo = _mm256_extracti128_si256(dists, 0);
            let dists_hi = _mm256_extracti128_si256(dists, 1);
            let arr: [u32; 4] = std::mem::transmute(dists_lo);
            let arr2: [u32; 4] = std::mem::transmute(dists_hi);

            for v in 0..8usize {
                let global_idx = bi * 8 + v;
                if global_idx >= end {
                    break;
                }
                let raw = match v {
                    0 => arr[0],
                    1 => arr[1],
                    2 => arr[2],
                    3 => arr[3],
                    4 => arr2[0],
                    5 => arr2[1],
                    6 => arr2[2],
                    _ => arr2[3],
                };
                let d = (raw as f32) * VECTOR_SCALE * VECTOR_SCALE;

                if d < best_d[KNN_K - 1] {
                    best_d[KNN_K - 1] = d;
                    best_id[KNN_K - 1] = global_idx as u32;
                    best_label[KNN_K - 1] = *labels.add(global_idx);
                    let mut j = KNN_K - 1;
                    while j > 0 && best_d[j] < best_d[j - 1] {
                        best_d.swap(j, j - 1);
                        best_id.swap(j, j - 1);
                        best_label.swap(j, j - 1);
                        j -= 1;
                    }
                }
            }

            bi += 1;
        }
    }
}

// ── Accuracy Test ─────────────────────────────────────────────────────
// Compares KNN with runtime NPROBE vs exact KNN (NPROBE=MAX_CENTROIDS).

use flate2::read::GzDecoder;
use serde::Deserialize;

#[derive(Deserialize)]
struct RefEntry {
    vector: [f32; 14],
    label: String,
}

/// Load references and run accuracy comparison.
/// Returns (total, mismatches, fp, fn).
pub fn accuracy_test(nprobe_test: usize, sample: usize) -> (usize, usize, usize, usize) {
    let ds = crate::data::dataset();

    let ref_path = std::env::var("REF_PATH")
        .unwrap_or_else(|_| "data/references.json.gz".to_string());
    eprintln!("[accuracy] loading {}...", ref_path);
    let file = std::fs::File::open(&ref_path).unwrap_or_else(|e| {
        let alt = format!("../../{}", ref_path);
        std::fs::File::open(&alt)
            .unwrap_or_else(|e2| panic!("can't open {} or {}: {}/{}", ref_path, alt, e, e2))
    });
    let gz = GzDecoder::new(std::io::BufReader::new(file));
    let entries: Vec<RefEntry> = serde_json::from_reader(gz).unwrap();
    eprintln!("[accuracy] {} entries loaded", entries.len());

    let total = entries.len();
    let step = if sample > 0 { total / sample } else { 1 };
    let queries: Vec<&[f32; 14]> =
        entries.iter().step_by(step).take(sample).map(|e| &e.vector).collect();
    let true_labels: Vec<u8> =
        entries.iter().step_by(step).take(sample).map(|e| {
            if e.label == "fraud" { 1u8 } else { 0u8 }
        }).collect();

    eprintln!(
        "[accuracy] testing {} queries with NPROBE={} vs exact...",
        queries.len(), nprobe_test
    );

    let mut mismatches = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;

    for (i, (&q, &true_y)) in queries.iter().zip(true_labels.iter()).enumerate() {
        let exact_fraud = knn5_fraud_count_nprobe(q, ds, MAX_CENTROIDS);
        let test_fraud = knn5_fraud_count_nprobe(q, ds, nprobe_test);

        if exact_fraud != test_fraud {
            mismatches += 1;
            if test_fraud >= 3 && exact_fraud < 3 {
                fp += 1;
            } else {
                fn_ += 1;
            }
        }
        if (i + 1) % 100 == 0 || i == queries.len() - 1 {
            eprintln!("[accuracy] {}/{} errors={} fp={} fn={}",
                i + 1, queries.len(), mismatches, fp, fn_);
        }
    }

    eprintln!(
        "[accuracy] DONE. NPROBE={:>3} | queries={} | errors={:>4} ({:.2}%) | fp={} | fn={}",
        nprobe_test, queries.len(),
        mismatches, mismatches as f64 / queries.len() as f64 * 100.0,
        fp, fn_
    );
    (queries.len(), mismatches, fp, fn_)
}
