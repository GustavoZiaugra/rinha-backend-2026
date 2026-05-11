use rinha_core::normalize::{self, TransactionPayload};
use rinha_core::vector::VectorSearch;
use rinha_core::dataset;
use serde::Deserialize;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct TestEntry {
    request: TransactionPayload,
    expected_approved: bool,
    expected_fraud_score: f32,
}

#[derive(Debug, Deserialize)]
struct TestData {
    stats: TestStats,
    entries: Vec<TestEntry>,
}

#[derive(Debug, Deserialize)]
struct TestStats { total: usize, fraud_count: usize, legit_count: usize }

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or_else(|| "data".to_string());
    let index_path = std::env::args().nth(2).unwrap_or_else(|| format!("{}/index.bin.gz", data_dir));
    println!("=== i16 Block Accuracy Test ===\n");
    let norm = dataset::load_normalization(&format!("{}/normalization.json", data_dir)).unwrap();
    let mcc_risk = dataset::load_mcc_risk(&format!("{}/mcc_risk.json", data_dir)).unwrap();
    let test_json = std::fs::read_to_string(&format!("{}/test-data.json", data_dir)).unwrap();
    let test_data: TestData = serde_json::from_str(&test_json).unwrap();

    let search = VectorSearch::load(&index_path);
    println!("Index loaded: {} blocks, {} centroids", search.total_blocks, search.k);

    let mut correct = 0usize;
    let mut fp = 0; let mut fn_ = 0;
    let max = test_data.entries.len().min(54100);
    let start = Instant::now();
    for (i, entry) in test_data.entries.iter().enumerate().take(max) {
        let query = normalize::normalize_transaction(&entry.request, &norm, &mcc_risk);
        let neighbors = search.search_with_probe(&query, 12);
        let fc = neighbors.iter().filter(|r| r.label == 1).count();
        let score = if neighbors.is_empty() { 0.0 } else { fc as f32 / neighbors.len() as f32 };
        let approved = score < 0.6;
        if approved == entry.expected_approved { correct += 1; }
        else if approved && !entry.expected_approved { fp += 1; }
        else { fn_ += 1; }
    }
    let elapsed = start.elapsed();
    println!("Results (i16 blocks): {}/{} correct ({:.1}%), FP={}, FN={}, time={:?}",
        correct, max, correct as f64 / max as f64 * 100.0, fp, fn_, elapsed);
}
