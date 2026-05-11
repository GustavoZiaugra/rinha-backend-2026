/// Build the IVF index offline and serialize to binary format.
/// Uses the same k-means implementation as VectorSearch::new()
/// so centroids and cluster assignments match exactly.
use std::io::Write;
use std::fs;
use flate2::write::GzEncoder;
use flate2::Compression;
use rinha_core::dataset;
use rinha_core::vector::{kmeans_f32, DIMS, f16_to_f32, N_CLUSTERS};

const SAMPLE_SIZE: usize = 50_000;
const KMEANS_ITERS: usize = 20;

fn scale_f32_to_i16(v: f32) -> i16 {
    (v * 10000.0).round() as i16
}

fn main() {
    println!("=== IVF Index Builder ===");

    let data_dir = std::env::args().nth(1).unwrap_or_else(|| "data".to_string());
    let output_path = std::env::args().nth(2).unwrap_or_else(|| "data/index.bin.gz".to_string());

    // Load vectors using streaming parser (avoids OOM on 284MB JSON)
    println!("Loading references (streaming)...");
    let (vectors_bits, labels) = dataset::load_references(
        &format!("{}/references.json.gz", data_dir)
    ).expect("load references");
    let count = labels.len();
    assert_eq!(vectors_bits.len() / DIMS, count);
    println!("Loaded {} vectors", count);

    // Sample for k-means (uses same sampling as VectorSearch::new)
    println!("K-means on {} sample...", SAMPLE_SIZE);
    let step = count / SAMPLE_SIZE;
    let mut sample = Vec::with_capacity(SAMPLE_SIZE * DIMS);
    for i in 0..SAMPLE_SIZE {
        let src = i * step;
        for d in 0..DIMS {
            sample.push(f16_to_f32(vectors_bits[src * DIMS + d]));
        }
    }
    let centroids = kmeans_f32(&sample, SAMPLE_SIZE, N_CLUSTERS, KMEANS_ITERS);
    let k = centroids.len();
    println!("K-means done ({} centroids)", k);

    // Assign all vectors to nearest centroid (same logic as new())
    println!("Assigning vectors to clusters...");
    let mut cluster_lists: Vec<Vec<(usize, u8)>> = (0..k).map(|_| Vec::new()).collect();
    for i in 0..count {
        let base = i * DIMS;
        let mut best = f32::MAX;
        let mut best_c = 0;
        for (ci, c) in centroids.iter().enumerate() {
            let mut sum = 0.0f32;
            for d in 0..DIMS {
                let diff = f16_to_f32(vectors_bits[base + d]) - c[d];
                sum += diff * diff;
            }
            if sum < best { best = sum; best_c = ci; }
        }
        cluster_lists[best_c].push((i, labels[i]));
    }

    // Debug cluster sizes
    for ci in 0..3.min(k) {
        let members = &cluster_lists[ci];
        println!("Cluster {}: {} members", ci, members.len());
        if !members.is_empty() {
            let (first_vi, _) = members[0];
            print!("  dims: ");
            for d in 0..DIMS {
                let val = f16_to_f32(vectors_bits[first_vi * DIMS + d]);
                print!("{:.4} ", val);
            }
            println!();
        }
    }

    // Build blocks: each block = 8 vectors in SOA layout
    println!("Building blocks...");
    let mut offsets = Vec::with_capacity(k + 1);
    let mut blocks: Vec<i16> = Vec::new();
    let mut flat_labels: Vec<u8> = Vec::new();

    offsets.push(0u32);
    for ci in 0..k {
        let members = &cluster_lists[ci];
        let n_blocks = (members.len() + 7) / 8;

        for bi in 0..n_blocks {
            for d in 0..DIMS {
                for j in 0..8 {
                    let idx = bi * 8 + j;
                    if idx < members.len() {
                        let (vi, _) = members[idx];
                        let val = f16_to_f32(vectors_bits[vi * DIMS + d]);
                        blocks.push(scale_f32_to_i16(val));
                    } else {
                        blocks.push(0i16); // padding
                    }
                }
            }
            // Labels for this block (padded to 8)
            for j in 0..8 {
                let idx = bi * 8 + j;
                if idx < members.len() {
                    flat_labels.push(members[idx].1);
                } else {
                    flat_labels.push(0); // padding
                }
            }
        }
        offsets.push(blocks.len() as u32 / 112);
    }

    // Build SOA centroids (f32, transposed: [dim0_all... dim1_all...])
    println!("Building SOA centroids...");
    let mut soa_centroids = Vec::with_capacity(DIMS * k);
    for d in 0..DIMS {
        for c in &centroids {
            soa_centroids.push(c[d]);
        }
    }

    // Serialize
    println!("Writing {}...", output_path);
    let file = fs::File::create(&output_path).unwrap();
    let mut gz = GzEncoder::new(file, Compression::best());

    // Magic
    gz.write_all(b"IVF1").unwrap();
    // Header: n, k, d
    gz.write_all(&(count as u32).to_le_bytes()).unwrap();
    gz.write_all(&(k as u32).to_le_bytes()).unwrap();
    gz.write_all(&(DIMS as u32).to_le_bytes()).unwrap();
    // Centroids (SOA f32)
    let cent_bytes: &[u8] = bytemuck::cast_slice(&soa_centroids);
    gz.write_all(cent_bytes).unwrap();
    // Offsets
    for &o in &offsets {
        gz.write_all(&o.to_le_bytes()).unwrap();
    }
    // Labels (padded per cluster)
    gz.write_all(&flat_labels).unwrap();
    // Blocks
    let block_bytes: &[u8] = bytemuck::cast_slice(&blocks);
    gz.write_all(block_bytes).unwrap();

    gz.finish().unwrap();
    println!("Done! index.bin.gz built.");
    println!("  {} centroids", k);
    println!("  {} total blocks", flat_labels.len() / 8);
    println!("  {} labels", flat_labels.len());
    println!("  {} i16 values", blocks.len());
    println!("  ~{} MB blocks", blocks.len() * 2 / 1048576);

    // ── SQ8 Quantized i8 vectors ──
    let i8_output = output_path.replace("index.bin.gz", "vectors_i8.bin.gz");
    println!("\nBuilding SQ8 i8 vectors -> {}...", i8_output);

    // 1. Compute per-dimension min/max from f16 vectors
    let mut qmin = [f32::MAX; DIMS];
    let mut qmax = [f32::MIN; DIMS];
    for i in 0..count {
        let base = i * DIMS;
        for d in 0..DIMS {
            let val = f16_to_f32(vectors_bits[base + d]);
            qmin[d] = qmin[d].min(val);
            qmax[d] = qmax[d].max(val);
        }
    }
    println!("  Per-dimension range:");
    for d in 0..DIMS.min(5) {
        println!("    dim[{}]: [{:.4}, {:.4}]", d, qmin[d], qmax[d]);
    }

    // 2. Quantize all vectors to i8 in cluster order
    // We need to build the i8 array in the SAME cluster order as the blocks
    let mut vectors_i8 = Vec::with_capacity(count * DIMS);
    for ci in 0..k {
        let members = &cluster_lists[ci];
        for &(vi, _) in members {
            let base = vi * DIMS;
            for d in 0..DIMS {
                let val = f16_to_f32(vectors_bits[base + d]);
                let range = qmax[d] - qmin[d];
                let i8_val = if range > 0.0 {
                    (((val - qmin[d]) / range * 255.0 - 128.0).round() as i32).clamp(-128, 127) as i8
                } else {
                    -128
                };
                vectors_i8.push(i8_val);
            }
        }
    }
    println!("  {} i8 vectors ({:.1} MB)", vectors_i8.len() / DIMS, vectors_i8.len() as f64 / 1048576.0);

    // 3. Write companion file
    let i8_file = fs::File::create(&i8_output).unwrap();
    let mut i8_gz = GzEncoder::new(i8_file, Compression::best());
    i8_gz.write_all(b"QI8V").unwrap();                  // magic
    i8_gz.write_all(&(count as u32).to_le_bytes()).unwrap(); // count
    i8_gz.write_all(&(DIMS as u32).to_le_bytes()).unwrap();  // dims
    // min per dim
    let min_bytes: &[u8] = bytemuck::cast_slice(&qmin);
    i8_gz.write_all(min_bytes).unwrap();
    // max per dim
    let max_bytes: &[u8] = bytemuck::cast_slice(&qmax);
    i8_gz.write_all(max_bytes).unwrap();
    // i8 vectors
    let i8_bytes: &[u8] = bytemuck::cast_slice(&vectors_i8);
    i8_gz.write_all(i8_bytes).unwrap();
    i8_gz.finish().unwrap();
    println!("Done! vectors_i8.bin.gz built.");
    println!("  ~{} MB compressed", std::fs::metadata(&i8_output).map(|m| m.len() / 1048576).unwrap_or(0));
}
