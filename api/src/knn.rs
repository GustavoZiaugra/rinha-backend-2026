use crate::data::{self, Dataset};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

const FAST_NPROBE: usize = 5;
const FULL_NPROBE: usize = 24;
const MAX_CENTROIDS: usize = 4096;
const VECTOR_SCALE: f32 = 0.0001;

pub fn knn5_fraud_count(query: &[f32; 14], ds: &Dataset) -> u8 {
    unsafe { knn5_ivf(query, ds) }
}

pub fn warmup() {
    let ds = data::dataset();
    let mut sink: u64 = 0;
    for v in ds.centroids.iter() {
        sink ^= v.to_bits() as u64;
    }
    let _ = sink;

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

#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_ivf(query: &[f32; 14], ds: &Dataset) -> u8 {
    let mut dists = [MaybeUninit::<f32>::uninit(); MAX_CENTROIDS];
    compute_centroid_dists(query, ds, &mut dists);

    let mut q_i16 = [0i16; 14];
    for d in 0..14usize {
        q_i16[d] = (query[d] / VECTOR_SCALE).round() as i16;
    }

    let fast_probes = top_n_from_dists::<FAST_NPROBE>(&dists, ds.k);
    let fast = scan_and_count(&fast_probes, ds, &q_i16);

    if fast != 2 && fast != 3 {
        return fast;
    }

    let full_probes = top_n_from_dists::<FULL_NPROBE>(&dists, ds.k);
    scan_and_count(&full_probes, ds, &q_i16)
}

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
            let s0 = _mm256_mul_ps(d0, d0);
            _mm256_storeu_ps(dp.add(ci), s0);
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

#[inline(always)]
unsafe fn top_n_from_dists<const N: usize>(
    dists: &[MaybeUninit<f32>; MAX_CENTROIDS],
    k: usize,
) -> [usize; N] {
    // Simple selection: keep N smallest in a max-heap-like array
    let dp = dists.as_ptr() as *const f32;
    let mut best = [(f32::MAX, 0usize); N];
    for i in 0..k {
        let d = *dp.add(i);
        if d < best[N - 1].0 {
            best[N - 1] = (d, i);
            // bubble up
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

#[target_feature(enable = "avx2,fma")]
unsafe fn scan_and_count(probes: &[usize], ds: &Dataset, q_i16: &[i16; 14]) -> u8 {
    // Keep top-5 nearest neighbors
    let mut best_d = [f32::MAX; 5];
    let mut best_id = [0u32; 5];
    let mut best_label = [0u8; 5];

    for &ci in probes {
        let start = ds.offsets[ci] as usize;
        let end = ds.offsets[ci + 1] as usize;
        let block_start = start / 8;
        let block_end = (end + 7) / 8;

        let blocks = ds.blocks.as_ptr();
        let labels = ds.labels.as_ptr();
        let block_stride = 14 * 8; // i16 per block

        for bi in block_start..block_end {
            let block_ptr = blocks.add(bi * block_stride);
            let mut dists8 = [0u64; 8];

            for d in 0..14usize {
                let qv = _mm256_set1_epi32(q_i16[d] as i32);
                let vals = _mm_loadu_si128(block_ptr.add(d * 8) as *const __m128i);
                let ve = _mm256_cvtepi16_epi32(vals);
                let diff = _mm256_sub_epi32(ve, qv);
                let sq = _mm256_mullo_epi32(diff, diff);
                let lo = _mm256_extracti128_si256(sq, 0);
                let hi = _mm256_extracti128_si256(sq, 1);
                let arr = std::mem::transmute::<__m128i, [u32; 4]>(lo);
                dists8[0] += arr[0] as u64;
                dists8[1] += arr[1] as u64;
                dists8[2] += arr[2] as u64;
                dists8[3] += arr[3] as u64;
                let arr2 = std::mem::transmute::<__m128i, [u32; 4]>(hi);
                dists8[4] += arr2[0] as u64;
                dists8[5] += arr2[1] as u64;
                dists8[6] += arr2[2] as u64;
                dists8[7] += arr2[3] as u64;
            }

            for v in 0..8usize {
                let global_idx = bi * 8 + v;
                if global_idx >= end {
                    break;
                }
                let d = (dists8[v] as f32) * VECTOR_SCALE * VECTOR_SCALE;

                // Insert into top-5
                if d < best_d[4] {
                    best_d[4] = d;
                    best_id[4] = global_idx as u32;
                    best_label[4] = *labels.add(global_idx);
                    let mut j = 4;
                    while j > 0 && best_d[j] < best_d[j - 1] {
                        best_d.swap(j, j - 1);
                        best_id.swap(j, j - 1);
                        best_label.swap(j, j - 1);
                        j -= 1;
                    }
                }
            }
        }
    }

    best_label.iter().filter(|&&l| l == 1).count() as u8
}
