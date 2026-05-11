use std::ptr;

pub const DIMS: usize = 14;
pub const N_CLUSTERS: usize = 512;
const KMEANS_ITERS: usize = 15;

/// IVF-based vector search using k-means clustering.
/// Vectors stored as f16 for fast SIMD using native F16C conversion.
pub struct VectorSearch {
    pub vectors: Vec<u16>,
    pub labels: Vec<u8>,
    pub centroids: Vec<u16>,
    /// Cluster offsets: (byte_offset_in_vectors, count)
    pub cluster_offsets: Vec<(usize, usize)>,
    pub k: usize,
    pub total_blocks: usize,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: u32,
    pub distance: f32,
    pub label: u8,
}

impl VectorSearch {
    pub fn load(path: &str) -> Self {
        use std::io::Read;
        let file = std::fs::File::open(path).unwrap();
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf).unwrap();

        let mut pos = 0;
        let magic = &buf[pos..pos + 4];
        pos += 4;
        assert_eq!(magic, b"IVF1", "Bad magic");

        let count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let k = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let dims = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        assert_eq!(dims, DIMS);

        // Centroids (SOA f32): k * DIMS floats
        let cent_bytes = k * DIMS * 4;
        let centroids_f32: Vec<f32> = bytemuck::cast_slice(&buf[pos..pos + cent_bytes]).to_vec();
        pos += cent_bytes;
        // Convert centroids from SOA to AOS
        let mut centroids = vec![0u16; k * DIMS];
        for ci in 0..k {
            for d in 0..DIMS {
                let f = centroids_f32[d * k + ci];
                centroids[ci * DIMS + d] = half::f16::from_f32(f).to_bits();
            }
        }

        // Offsets: (k+1) u32
        let off_bytes = (k + 1) * 4;
        let offsets_raw: &[u32] = bytemuck::cast_slice(&buf[pos..pos + off_bytes]);
        pos += off_bytes;

        let total_blocks_stored = offsets_raw[k] as usize;
        let label_count = total_blocks_stored * 8;
        let labels_raw = buf[pos..pos + label_count].to_vec();
        pos += label_count;

        // Blocks: i16 SOA per block [dim0*8, dim1*8, ..., dim13*8]
        let block_i16: &[i16] = bytemuck::cast_slice(&buf[pos..]);

        // De-interleave blocks into f16 vectors (AOS)
        let mut vectors = vec![0u16; count * DIMS];
        let mut reordered_labels = vec![0u8; count];
        let mut vi = 0usize;
        let mut cluster_offsets = Vec::with_capacity(k);
        let mut start_pos = 0usize;

        for ci in 0..k {
            let block_start = offsets_raw[ci] as usize;
            let block_end = offsets_raw[ci + 1] as usize;
            let n_blocks = block_end - block_start;

            for bi in 0..n_blocks {
                let base_block = (block_start + bi) * DIMS * 8;
                for slot in 0..8 {
                    let global_idx = block_start * 8 + bi * 8 + slot;
                    if global_idx >= count { break; }
                    if vi >= count { break; }

                    for d in 0..DIMS {
                        let i16_val = block_i16[base_block + d * 8 + slot];
                        let f32_val = i16_val as f32 / 10000.0;
                        vectors[vi * DIMS + d] = half::f16::from_f32(f32_val).to_bits();
                    }
                    reordered_labels[vi] = labels_raw[global_idx];
                    vi += 1;
                }
            }
            cluster_offsets.push((start_pos * DIMS, vi - start_pos));
            start_pos = vi;
        }

        Self {
            vectors,
            labels: reordered_labels,
            centroids,
            cluster_offsets,
            k,
            total_blocks: total_blocks_stored,
            count,
        }
    }

    pub fn search(&self, query: &[f32; DIMS], _k: usize) -> Vec<SearchResult> {
        self.search_with_probe(query, 4)
    }

    pub fn search_with_probe(&self, query: &[f32; DIMS], n_probe: usize) -> Vec<SearchResult> {
        if self.count == 0 { return Vec::new(); }
        let closest = find_closest_centroids(query, &self.centroids, self.k, n_probe);
        self.search_clusters(query, &closest)
    }

    fn search_clusters(&self, query: &[f32; DIMS], clusters: &[usize]) -> Vec<SearchResult> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        { self.search_simd(query, clusters) }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        { self.search_scalar(query, clusters) }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn search_simd(&self, query: &[f32; DIMS], clusters: &[usize]) -> Vec<SearchResult> {
        use std::arch::x86_64::*;
        let q_lo = unsafe { _mm256_set_ps(query[7], query[6], query[5], query[4], query[3], query[2], query[1], query[0]) };
        let q_hi = unsafe { _mm256_set_ps(0.0, 0.0, query[13], query[12], query[11], query[10], query[9], query[8]) };

        let vecs_ptr = self.vectors.as_ptr();
        let labels_ptr = self.labels.as_ptr();

        let mut top = [(f32::MAX, 0u32, 0u8); 5];
        let mut filled = 0usize;
        let mut widx = 0usize;

        for &c in clusters {
            let (byte_off, cnt) = self.cluster_offsets[c];
            let start = byte_off / DIMS;
            for j in 0..cnt {
                let idx = start + j;
                let dist = unsafe { simd_distance(vecs_ptr, idx, q_lo, q_hi) };
                let label = unsafe { *labels_ptr.add(idx) };
                if filled < 5 {
                    top[filled] = (dist, idx as u32, label);
                    filled += 1;
                    if filled == 5 { widx = (0..5).max_by(|&a, &b| top[a].0.total_cmp(&top[b].0)).unwrap(); }
                } else if dist < top[widx].0 {
                    top[widx] = (dist, idx as u32, label);
                    widx = (0..5).max_by(|&a, &b| top[a].0.total_cmp(&top[b].0)).unwrap();
                }
            }
        }

        let mut order: [usize; 5] = [0, 1, 2, 3, 4];
        order.sort_by(|&a, &b| top[a].0.total_cmp(&top[b].0));
        (0..filled).map(|i| { let (d, id, lb) = top[order[i]]; SearchResult { node_id: id, distance: d, label: lb } }).collect()
    }

    fn search_scalar(&self, query: &[f32; DIMS], clusters: &[usize]) -> Vec<SearchResult> {
        let mut top = [(f32::MAX, 0u32, 0u8); 5];
        let mut filled = 0usize;
        let mut widx = 0usize;
        for &c in clusters {
            let (byte_off, cnt) = self.cluster_offsets[c];
            let start = byte_off / DIMS;
            for j in 0..cnt {
                let idx = start + j;
                let mut sum = 0.0f32;
                let base = idx * DIMS;
                for d in 0..DIMS { let diff = query[d] - half::f16::from_bits(self.vectors[base + d]).to_f32(); sum += diff * diff; }
                let label = self.labels[idx];
                if filled < 5 { top[filled] = (sum, idx as u32, label); filled += 1; if filled == 5 { widx = (0..5).max_by(|&a, &b| top[a].0.total_cmp(&top[b].0)).unwrap(); } }
                else if sum < top[widx].0 { top[widx] = (sum, idx as u32, label); widx = (0..5).max_by(|&a, &b| top[a].0.total_cmp(&top[b].0)).unwrap(); }
            }
        }
        let mut order: [usize; 5] = [0, 1, 2, 3, 4];
        order.sort_by(|&a, &b| top[a].0.total_cmp(&top[b].0));
        (0..filled).map(|i| { let (d, id, lb) = top[order[i]]; SearchResult { node_id: id, distance: d, label: lb } }).collect()
    }

    pub fn memory_usage(&self) -> usize {
        self.vectors.len() * 2 + self.labels.len() + self.centroids.len() * 2 + self.cluster_offsets.len() * 16
    }
}

fn find_closest_centroids(query: &[f32; DIMS], centroids: &[u16], k: usize, n: usize) -> Vec<usize> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    { find_closest_centroids_simd(query, centroids, k, n) }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    { find_closest_centroids_scalar(query, centroids, k, n) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn find_closest_centroids_simd(query: &[f32; DIMS], centroids: &[u16], k: usize, n: usize) -> Vec<usize> {
    use std::arch::x86_64::*;
    let q_lo = unsafe { _mm256_set_ps(query[7], query[6], query[5], query[4], query[3], query[2], query[1], query[0]) };
    let q_hi = unsafe { _mm256_set_ps(0.0, 0.0, query[13], query[12], query[11], query[10], query[9], query[8]) };
    let cp = centroids.as_ptr();
    let mut dists = [(f32::MAX, 0usize); 512];
    for c in 0..k { dists[c] = (unsafe { simd_distance(cp, c, q_lo, q_hi) }, c); }
    let dists_slice = &mut dists[..k];
    if n < k { dists_slice.select_nth_unstable_by(n, |a, b| a.0.total_cmp(&b.0)); }
    dists_slice[..n].sort_by(|a, b| a.0.total_cmp(&b.0));
    dists_slice[..n].iter().map(|&(_, c)| c).collect()
}

fn find_closest_centroids_scalar(query: &[f32; DIMS], centroids: &[u16], k: usize, n: usize) -> Vec<usize> {
    let mut dists = [(f32::MAX, 0usize); 512];
    for c in 0..k {
        let mut sum = 0.0f32;
        for d in 0..DIMS { let diff = query[d] - half::f16::from_bits(centroids[c * DIMS + d]).to_f32(); sum += diff * diff; }
        dists[c] = (sum, c);
    }
    let dists_slice = &mut dists[..k];
    if n < k { dists_slice.select_nth_unstable_by(n, |a, b| a.0.total_cmp(&b.0)); }
    dists_slice[..n].sort_by(|a, b| a.0.total_cmp(&b.0));
    dists_slice[..n].iter().map(|&(_, c)| c).collect()
}

pub fn f16_to_f32(bits: u16) -> f32 { half::f16::from_bits(bits).to_f32() }

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn simd_distance(vecs: *const u16, idx: usize, q_lo: std::arch::x86_64::__m256, q_hi: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let ptr = vecs.add(idx * DIMS);
    let v_lo = _mm256_cvtph_ps(_mm_loadu_si128(ptr as *const __m128i));
    let v_hi = _mm256_mul_ps(
        _mm256_cvtph_ps(_mm_loadu_si128(ptr.add(8) as *const __m128i)),
        _mm256_set_ps(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    );
    let zero = _mm256_setzero_ps();
    let sq_l = _mm256_fmadd_ps(_mm256_sub_ps(q_lo, v_lo), _mm256_sub_ps(q_lo, v_lo), zero);
    let sq_h = _mm256_fmadd_ps(_mm256_sub_ps(q_hi, v_hi), _mm256_sub_ps(q_hi, v_hi), zero);
    let s_l = _mm256_hadd_ps(_mm256_hadd_ps(sq_l, sq_l), sq_l);
    let s_h = _mm256_hadd_ps(_mm256_hadd_ps(sq_h, sq_h), sq_h);
    _mm_cvtss_f32(_mm_add_ss(
        _mm_add_ss(_mm256_castps256_ps128(s_l), _mm256_extractf128_ps(s_l, 1)),
        _mm_add_ss(_mm256_castps256_ps128(s_h), _mm256_extractf128_ps(s_h, 1)),
    ))
}

pub fn kmeans_f32(sample: &[f32], n: usize, k: usize, max_iter: usize) -> Vec<[f32; DIMS]> {
    let k = k.min(if n > 100 { n / 2 } else { n }).max(1);
    let mut rng = 42u64;
    let mut rng_f32 = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as f32 / (u32::MAX as f32)
    };

    let mut centroids: Vec<[f32; DIMS]> = Vec::with_capacity(k);
    let mut chosen = vec![false; n];
    let first = (rng_f32() * n as f32) as usize % n;
    let mut c0 = [0.0f32; DIMS];
    for d in 0..DIMS { c0[d] = sample[first * DIMS + d]; }
    centroids.push(c0);
    chosen[first] = true;

    while centroids.len() < k {
        let mut dists = Vec::with_capacity(n);
        for i in 0..n {
            let mut min_d = f32::MAX;
            for c in &centroids {
                let mut sum = 0.0f32;
                for d in 0..DIMS { let diff = sample[i * DIMS + d] - c[d]; sum += diff * diff; }
                if sum < min_d { min_d = sum; }
            }
            dists.push(min_d);
        }
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            while centroids.len() < k {
                let ri = (rng_f32() * n as f32) as usize % n;
                let mut c = [0.0f32; DIMS];
                for d in 0..DIMS { c[d] = sample[ri * DIMS + d]; }
                centroids.push(c);
            }
            break;
        }
        let threshold = rng_f32() * total;
        let mut cum = 0.0f32;
        for i in 0..n {
            cum += dists[i];
            if cum >= threshold && !chosen[i] {
                let mut c = [0.0f32; DIMS]; for d in 0..DIMS { c[d] = sample[i * DIMS + d]; }
                centroids.push(c);
                chosen[i] = true;
                break;
            }
        }
    }

    let mut assignments = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let mut best_d = f32::MAX; let mut best_c = 0;
            for (ci, c) in centroids.iter().enumerate() {
                let mut sum = 0.0f32;
                for d in 0..DIMS { let diff = sample[i * DIMS + d] - c[d]; sum += diff * diff; }
                if sum < best_d { best_d = sum; best_c = ci; }
            }
            if assignments[i] != best_c { changed = true; }
            assignments[i] = best_c;
        }
        if !changed { break; }
        let mut sums = vec![[0.0f32; DIMS]; k];
        let mut cnts = vec![0usize; k];
        for i in 0..n { let c = assignments[i]; for d in 0..DIMS { sums[c][d] += sample[i * DIMS + d]; } cnts[c] += 1; }
        for c in 0..k { if cnts[c] > 0 { for d in 0..DIMS { sums[c][d] /= cnts[c] as f32; } } }
        centroids = sums;
    }
    centroids
}
