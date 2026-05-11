use rinha_core::dataset;
use rinha_core::normalize::{self, TransactionPayload};
use rinha_core::vector::VectorSearch;
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
struct TestStats {
    total: usize,
    fraud_count: usize,
    legit_count: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.get(1).map(|s| s.as_str()).unwrap_or("data");
    let index_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| format!("{}/index.bin.gz", data_dir));
    let test_data_path = format!("{}/test-data.json", data_dir);

    println!("=== Rinha Backend 2026 - Test Suite ===\n");

    let norm = dataset::load_normalization(&format!("{}/normalization.json", data_dir))
        .expect("load normalization");
    let mcc_risk =
        dataset::load_mcc_risk(&format!("{}/mcc_risk.json", data_dir)).expect("load mcc_risk");
    println!("Normalization: {:?}", norm);
    println!("MCC Risk categories: {}\n", mcc_risk.len());

    // Load index
    let load_start = Instant::now();
    let search = VectorSearch::load(&index_path);
    println!(
        "Index loaded in {:?} (memory: ~{} MB, {} blocks)\n",
        load_start.elapsed(),
        search.memory_usage() / 1048576,
        search.total_blocks
    );

    // Load test data
    let test_json = std::fs::read_to_string(&test_data_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", test_data_path));
    let test_data: TestData = serde_json::from_str(&test_json).expect("Failed to parse test data");
    println!(
        "Test data: {} entries (fraud: {}, legit: {})\n",
        test_data.stats.total, test_data.stats.fraud_count, test_data.stats.legit_count
    );

    // Run tests
    let test_start = Instant::now();
    let mut correct = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;
    let mut total_score_diff = 0.0f32;
    let mut total_time = 0.0f32;

    let max_tests = test_data.entries.len();

    for (i, entry) in test_data.entries.iter().enumerate().take(max_tests) {
        let req_start = Instant::now();
        let query = normalize::normalize_transaction(&entry.request, &norm, &mcc_risk);
        let neighbors = search.search(&query, 5);
        let req_elapsed = req_start.elapsed();

        let fraud_count = neighbors.iter().filter(|r| r.label == 1).count();
        let fraud_score = if neighbors.is_empty() {
            0.0
        } else {
            fraud_count as f32 / neighbors.len() as f32
        };
        let approved = fraud_score < 0.6;

        total_time += req_elapsed.as_secs_f32() * 1000.0;
        total_score_diff += (fraud_score - entry.expected_fraud_score).abs();

        if approved == entry.expected_approved {
            correct += 1;
        } else if approved && !entry.expected_approved {
            false_positives += 1;
        } else {
            false_negatives += 1;
        }

        if i < 5 || (approved != entry.expected_approved && i < 100) {
            let status = if approved == entry.expected_approved {
                "✓"
            } else {
                "✗"
            };
            println!(
                "  {} tx-{}: expected={}, got={}, score={:.3} (exp={:.3}) [{:.3}ms]",
                status,
                i,
                entry.expected_approved,
                approved,
                fraud_score,
                entry.expected_fraud_score,
                req_elapsed.as_secs_f32() * 1000.0
            );
        }
    }

    let avg_time = total_time / max_tests as f32;
    let accuracy = correct as f64 / max_tests as f64 * 100.0;
    println!("\n=== Results ===");
    println!("  Total tested: {}", max_tests);
    println!("  Correct: {} ({:.1}%)", correct, accuracy);
    println!("  False Positives: {}", false_positives);
    println!("  False Negatives: {}", false_negatives);
    println!(
        "  Avg score diff: {:.4}",
        total_score_diff / max_tests as f32
    );
    println!("  Avg query time: {:.3}ms", avg_time);
    println!("  Total test time: {:?}", test_start.elapsed());
}
