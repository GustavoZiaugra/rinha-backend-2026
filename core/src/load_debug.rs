use rinha_core::vector::VectorSearch;
use std::time::Instant;

fn main() {
    let data_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/zig/projects/rinha-backend-2026/data".to_string());
    let index_path = format!("{}/index.bin.gz", data_dir);

    println!("=== Loading index ===\n");
    let load_start = Instant::now();
    let search = VectorSearch::load(&index_path);
    println!(
        "Loaded in {:?}: {} vectors, {} clusters, {} blocks, ~{} MB",
        load_start.elapsed(),
        search.count,
        search.k,
        search.total_blocks,
        search.memory_usage() / 1048576
    );

    // Quick search test
    let query = [0.1f32; 14];
    let r = search.search(&query, 5);
    println!("\nQuick search (query=[0.1; 14]): {:?}", r.len());
    for res in &r {
        println!("  node={}, dist={:.4}, label={}", res.node_id, res.distance, res.label);
    }
}
