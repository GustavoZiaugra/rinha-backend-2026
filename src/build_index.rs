use flate2::read::GzDecoder;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

const K: usize = 4096;
const D: usize = 14;
const N_ITER: usize = 25;

fn main() {
    let t0 = Instant::now();

    let data_dir = std::env::args().nth(1).unwrap_or_else(|| "data".to_string());
    let output_path = std::env::args().nth(2).unwrap_or_else(|| format!("{}/index.bin.gz", data_dir));

    eprintln!("loading dataset...");
    let (vectors, labels) = load_dataset(&format!("{}/references.json.gz", data_dir));
    let n = vectors.len();
    eprintln!("  {} vectors in {:?}", n, t0.elapsed());

    eprintln!("kmeans++ init...");
    let t1 = Instant::now();
    let mut centroids = kmeans_plus_plus_init(&vectors, K);
    eprintln!("  done in {:?}", t1.elapsed());

    eprintln!("lloyd iterations...");
    let mut assignments = vec![0u16; n];
    for iter in 0..N_ITER {
        let t = Instant::now();
        let changed = assign_all(&vectors, &centroids, &mut assignments);
        update_centroids(&vectors, &assignments, &mut centroids);
        eprintln!(
            "  iter {:2}: {:5.2}% changed in {:?}",
            iter + 1,
            changed as f64 / n as f64 * 100.0,
            t.elapsed()
        );
        if changed * 1000 < n {
            break;
        }
    }

    eprintln!("writing index...");
    let t2 = Instant::now();
    write_index(&vectors, &labels, &assignments, &centroids, n, &output_path);
    eprintln!("  written in {:?}", t2.elapsed());
    eprintln!("total: {:?}", t0.elapsed());
}

#[derive(Deserialize)]
struct RefEntry {
    vector: [f32; 14],
    label: String,
}

fn load_dataset(path: &str) -> (Vec<[f32; 14]>, Vec<u8>) {
    let file = File::open(path).unwrap();
    let gz = GzDecoder::new(std::io::BufReader::new(file));
    let entries: Vec<RefEntry> = serde_json::from_reader(gz).unwrap();
    let vecs: Vec<[f32; 14]> = entries.iter().map(|e| e.vector).collect();
    let lbls: Vec<u8> = entries.iter().map(|e| if e.label == "fraud" { 1 } else { 0 }).collect();
    (vecs, lbls)
}

fn kmeans_plus_plus_init(vectors: &[[f32; 14]], k: usize) -> Vec<[f32; 14]> {
    let n = vectors.len();
    let mut rng = 0xdeadbeef_cafebabe_u64;
    let mut rng_next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    };

    let mut centroids = Vec::with_capacity(k);
    let mut chosen = vec![false; n];
    let first = (rng_next() >> 33) as usize % n;
    centroids.push(vectors[first]);
    chosen[first] = true;

    let mut dists = vec![f32::MAX; n];
    for _ in 1..k {
        let last = *centroids.last().unwrap();
        let mut total = 0.0f64;
        for i in 0..n {
            if chosen[i] {
                dists[i] = 0.0;
                continue;
            }
            let d = dist2(&vectors[i], &last);
            if d < dists[i] {
                dists[i] = d;
            }
            total += dists[i] as f64;
        }
        let thresh = (rng_next() >> 11) as f64 / (1u64 << 53) as f64 * total;
        let mut cum = 0.0f64;
        let mut pick = 0usize;
        for i in 0..n {
            cum += dists[i] as f64;
            if cum >= thresh {
                pick = i;
                break;
            }
        }
        centroids.push(vectors[pick]);
        chosen[pick] = true;
    }
    centroids
}

fn assign_all(vectors: &[[f32; 14]], centroids: &[[f32; 14]], assignments: &mut [u16]) -> usize {
    let n = vectors.len();
    let k = centroids.len();
    let mut changed = 0usize;
    for i in 0..n {
        let mut best = f32::MAX;
        let mut best_c = 0usize;
        for c in 0..k {
            let d = dist2(&vectors[i], &centroids[c]);
            if d < best {
                best = d;
                best_c = c;
            }
        }
        if assignments[i] as usize != best_c {
            changed += 1;
            assignments[i] = best_c as u16;
        }
    }
    changed
}

fn update_centroids(vectors: &[[f32; 14]], assignments: &[u16], centroids: &mut [[f32; 14]]) {
    let k = centroids.len();
    let mut counts = vec![0usize; k];
    for c in centroids.iter_mut() {
        *c = [0.0f32; 14];
    }
    for (i, &a) in assignments.iter().enumerate() {
        let a = a as usize;
        counts[a] += 1;
        for d in 0..14 {
            centroids[a][d] += vectors[i][d];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            for d in 0..14 {
                centroids[c][d] *= inv;
            }
        }
    }
}

fn dist2(a: &[f32; 14], b: &[f32; 14]) -> f32 {
    let mut s = 0.0f32;
    for d in 0..14 {
        let diff = a[d] - b[d];
        s += diff * diff;
    }
    s
}

fn write_index(
    vectors: &[[f32; 14]],
    labels: &[u8],
    assignments: &[u16],
    centroids: &[[f32; 14]],
    n: usize,
    path: &str,
) {
    let k = K;
    let mut cluster_lists: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    for (i, &a) in assignments.iter().enumerate() {
        cluster_lists[a as usize].push(i);
    }

    // Pad each cluster to multiple of 8
    let mut padded_labels = Vec::new();
    let mut offsets = vec![0u32; k + 1];
    let mut all_blocks: Vec<i16> = Vec::new();

    for c in 0..k {
        offsets[c] = padded_labels.len() as u32;
        let members = &cluster_lists[c];
        let mut clabels: Vec<u8> = members.iter().map(|&i| labels[i]).collect();
        // Pad to multiple of 8
        while clabels.len() % 8 != 0 {
            clabels.push(0);
        }
        padded_labels.extend_from_slice(&clabels);

        // Build blocks: each block = 8 vectors, dim-major [d][8] i16
        let padded_count = clabels.len();
        let num_blocks = padded_count / 8;
        for b in 0..num_blocks {
            let mut block = [[0i16; 8]; 14];
            for v in 0..8 {
                let orig_idx = members.get(b * 8 + v);
                if let Some(&idx) = orig_idx {
                    for d in 0..14 {
                        block[d][v] = (vectors[idx][d] * 10000.0).round() as i16;
                    }
                }
            }
            // Flatten block to dim-major
            for d in 0..14 {
                for v in 0..8 {
                    all_blocks.push(block[d][v]);
                }
            }
        }
    }
    offsets[k] = padded_labels.len() as u32;

    let file = File::create(path).unwrap();
    let mut gz = flate2::write::GzEncoder::new(BufWriter::new(file), flate2::Compression::default());

    gz.write_all(b"IVF1").unwrap();
    gz.write_all(&(n as u32).to_le_bytes()).unwrap();
    gz.write_all(&(k as u32).to_le_bytes()).unwrap();
    gz.write_all(&(14u32).to_le_bytes()).unwrap();

    // Centroids SOA f32
    let mut soa = vec![0f32; k];
    for d in 0..14 {
        for c in 0..k {
            soa[c] = centroids[c][d];
        }
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(soa.as_ptr() as *const u8, soa.len() * 4) };
        gz.write_all(bytes).unwrap();
    }

    // Offsets
    let off_bytes: &[u8] = unsafe { std::slice::from_raw_parts(offsets.as_ptr() as *const u8, offsets.len() * 4) };
    gz.write_all(off_bytes).unwrap();

    // Labels
    gz.write_all(&padded_labels).unwrap();

    // Blocks
    let block_bytes: &[u8] = unsafe { std::slice::from_raw_parts(all_blocks.as_ptr() as *const u8, all_blocks.len() * 2) };
    gz.write_all(block_bytes).unwrap();

    gz.finish().unwrap();
}
