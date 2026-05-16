use crate::data::{self, Dataset};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

/// Two-tier search: fast probe first, only do full probe when borderline
const FAST_NPROBE: usize = 5;
const FULL_NPROBE: usize = 24;
const MAX_CENTROIDS: usize = 4096;
const VECTOR_SCALE: f32 = 0.0001;

pub fn knn5_fraud_count(query: &[f32; 14], ds: &Dataset) -> u8 {
    unsafe { knn5_ivf(query, ds) }
}

/// Warm up: XOR all centroids + offsets to force page residency,
/// then run 500 synthetic queries to warm CPU caches.
pub fn warmup() {
    let ds = data::dataset();

    let mut sink: u64 = 0;

    // XOR centroids to touch every cache line
    for v in ds.centroids.iter() {
        sink ^= v.to_bits() as u64;
    }

    // XOR offsets too
    for &v in ds.offsets.iter() {
        sink ^= v as u64;
    }

    let _ = sink;

    // 500 warmup queries with LCG PRNG
    let mut state = 0x12345678u32;
    for _ in 0..500 {
        let mut q = [0.0f32; 14];
        for v in q.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (state >> 8) as f32 / (1u32 << 24) as f32;
        }
        let _ = knn5_fraud_count(&q, ds);
    }
}

// ---------------------------------------------------------------------------
// Top-level KNN: two-tier IVF search
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_ivf(query: &[f32; 14], ds: &Dataset) -> u8 {
    let mut dists = [MaybeUninit::<f32>::uninit(); MAX_CENTROIDS];
    compute_centroid_dists(query, ds, &mut dists);

    // Broadcast query per dimension for SIMD block scan
    let mut q_vecs = [_mm256_setzero_ps(); 14];
    for d in 0..14usize {
        q_vecs[d] = _mm256_set1_ps(query[d]);
    }

    // Tier 1: fast probe
    let fast_probes = top_n_from_dists::<FAST_NPROBE>(&dists, ds.k);
    let fast = scan_and_count(&fast_probes, ds, &q_vecs);

    // If not in {2, 3}, we're confident — skip full probe
    if fast != 2 && fast != 3 {
        return fast;
    }

    // Tier 2: full probe for borderline cases
    let full_probes = top_n_from_dists::<FULL_NPROBE>(&dists, ds.k);
    scan_and_count(&full_probes, ds, &q_vecs)
}

// ---------------------------------------------------------------------------
// Centroid distance (SOA f32 → squared Euclidean, AVX2 FMA + 16-wide)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma")]
unsafe fn compute_centroid_dists(
    query: &[f32; 14],
    ds: &Dataset,
    dists: &mut [MaybeUninit<f32>; MAX_CENTROIDS],
) {
    let k = ds.k;
    let cp = ds.centroids.as_ptr();
    let dp = dists.as_mut_ptr() as *mut f32;

    // Dimension 0 (scalar init)
    {
        let qd = _mm256_set1_ps(query[0]);
        let mut ci = 0usize;
        while ci + 16 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci)), qd);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(ci + 8)), qd);
            _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_mul_ps(d1, d1));
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

    // Dimensions 1..13 (accumulate with FMA)
    for dim in 1..14usize {
        let qd = _mm256_set1_ps(query[dim]);
        let base = dim * k;
        let mut ci = 0usize;
        while ci + 16 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(base + ci)), qd);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(base + ci + 8)), qd);
            let a0 = _mm256_loadu_ps(dp.add(ci));
            let a1 = _mm256_loadu_ps(dp.add(ci + 8));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(d0, d0, a0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_fmadd_ps(d1, d1, a1));
            ci += 16;
        }
        while ci + 8 <= k {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(cp.add(base + ci)), qd);
            let a0 = _mm256_loadu_ps(dp.add(ci));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(d0, d0, a0));
            ci += 8;
        }
        while ci < k {
            let diff = *cp.add(base + ci) - query[dim];
            *dp.add(ci) += diff * diff;
            ci += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Select top-N centroids by distance (AVX2 mask-scan)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma")]
unsafe fn top_n_from_dists<const N: usize>(
    dists: &[MaybeUninit<f32>; MAX_CENTROIDS],
    k: usize,
) -> [usize; N] {
    let mut top_dists = [f32::INFINITY; N];
    let mut top_idx = [0usize; N];
    let dp = dists.as_ptr() as *const f32;
    let mut ci = 0usize;

    // Process 8 centroids at a time with AVX2
    while ci + 8 <= k {
        let d8 = _mm256_loadu_ps(dp.add(ci));
        let mask = _mm256_movemask_ps(_mm256_cmp_ps(
            d8,
            _mm256_set1_ps(top_dists[N - 1]),
            _CMP_LT_OQ,
        )) as u32;

        if mask != 0 {
            let mut buf = [0.0f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), d8);
            let mut m = mask;
            while m != 0 {
                let s = m.trailing_zeros() as usize;
                m &= m - 1;
                let di = buf[s];
                if di < top_dists[N - 1] {
                    let pos = top_dists.partition_point(|&x| x < di);
                    top_dists[pos..N].rotate_right(1);
                    top_dists[pos] = di;
                    top_idx[pos..N].rotate_right(1);
                    top_idx[pos] = ci + s;
                }
            }
        }
        ci += 8;
    }

    // Tail: remaining centroids (scalar)
    while ci < k {
        let di = *dp.add(ci);
        if di < top_dists[N - 1] {
            let pos = top_dists.partition_point(|&x| x < di);
            top_dists[pos..N].rotate_right(1);
            top_dists[pos] = di;
            top_idx[pos..N].rotate_right(1);
            top_idx[pos] = ci;
        }
        ci += 1;
    }

    top_idx
}

// ---------------------------------------------------------------------------
// Scan clusters: accumulate distances via FMA, early-terminate, prefetch
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma")]
unsafe fn scan_and_count(probes: &[usize], ds: &Dataset, q_vecs: &[__m256; 14]) -> u8 {
    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;
    let blocks_ptr = ds.blocks.as_ptr();
    let labels_ptr = ds.labels.as_ptr();

    for &ci in probes {
        // Convert label offsets to block offsets (both are multiples of 8)
        let start = *ds.offsets.as_ptr().add(ci) as usize / 8;
        let end = *ds.offsets.as_ptr().add(ci + 1) as usize / 8;
        scan_blocks(
            q_vecs,
            blocks_ptr,
            labels_ptr,
            start,
            end,
            &mut top,
            &mut worst_idx,
        );
    }

    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

/// Process one cluster's blocks: for each block of 8 vectors, compute
/// squared Euclidean distance in f32 space using AVX2 FMA, with
/// early termination after 8 dimensions and prefetch 8 blocks ahead.
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_blocks(
    q_vecs: &[__m256; 14],
    blocks_ptr: *const i16,
    labels_ptr: *const u8,
    start_block: usize,
    end_block: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    let scale = _mm256_set1_ps(VECTOR_SCALE);

    // Process 2 dimensions at a time using interleaved accumulators.
    // acc0 accumulates even dims, acc1 accumulates odd dims.
    // After all 7 pairs, sum acc0+acc1 for total per-vector distance.
    macro_rules! dim_pair {
        ($acc0:expr, $acc1:expr, $bb:expr, $d:expr) => {{
            let r0 = _mm_loadu_si128(blocks_ptr.add($bb + $d * 8) as *const __m128i);
            let r1 = _mm_loadu_si128(blocks_ptr.add($bb + ($d + 1) * 8) as *const __m128i);
            let v0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r0)), scale);
            let v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r1)), scale);
            let d0 = _mm256_sub_ps(v0, q_vecs[$d]);
            let d1 = _mm256_sub_ps(v1, q_vecs[$d + 1]);
            $acc0 = _mm256_fmadd_ps(d0, d0, $acc0);
            $acc1 = _mm256_fmadd_ps(d1, d1, $acc1);
        }};
    }

    'block: for block_i in start_block..end_block {
        // Prefetch 8 blocks ahead to overlap memory latency
        let prefetch_block = block_i + 8;
        if prefetch_block < end_block {
            // Each block = 112 i16s = 224 bytes
            // Prefetch both halves
            _mm_prefetch(
                blocks_ptr.add(prefetch_block * 112) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                blocks_ptr.add(prefetch_block * 112 + 56) as *const i8,
                _MM_HINT_T0,
            );
        }

        let bb = block_i * 112; // i16 offset for this block (14 dims * 8 vecs)
        let threshold = _mm256_set1_ps(top[*worst_idx].0);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // Dim pairs 0-7 (4 pairs)
        dim_pair!(acc0, acc1, bb, 0);
        dim_pair!(acc0, acc1, bb, 2);
        dim_pair!(acc0, acc1, bb, 4);
        dim_pair!(acc0, acc1, bb, 6);

        // Check partial sum: if no vector passes threshold after 8 dims, skip
        let partial = _mm256_add_ps(acc0, acc1);
        if _mm256_movemask_ps(_mm256_cmp_ps(partial, threshold, _CMP_LT_OQ)) == 0 {
            continue 'block;
        }

        // Dim pairs 8-13 (3 pairs)
        dim_pair!(acc0, acc1, bb, 8);
        dim_pair!(acc0, acc1, bb, 10);
        dim_pair!(acc0, acc1, bb, 12);

        let acc = _mm256_add_ps(acc0, acc1);
        let mut mask = _mm256_movemask_ps(_mm256_cmp_ps(acc, threshold, _CMP_LT_OQ)) as u32;
        if mask == 0 {
            continue;
        }

        // Unpack surviving vectors and update top-K
        let mut dists_buf = [0.0f32; 8];
        _mm256_storeu_ps(dists_buf.as_mut_ptr(), acc);
        let label_base = block_i * 8;
        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let di = dists_buf[slot];
            if di < top[*worst_idx].0 {
                top[*worst_idx] = (di, *labels_ptr.add(label_base + slot));
                // Find new worst (simplified linear scan over 5 elements — tiny)
                let mut wi = 0;
                let mut wv = top[0].0;
                for j in 1..5 {
                    if top[j].0 > wv {
                        wv = top[j].0;
                        wi = j;
                    }
                }
                *worst_idx = wi;
            }
        }
    }
}
