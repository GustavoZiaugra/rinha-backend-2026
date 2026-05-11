use flate2::read::GzDecoder;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};

/// Normalization constants from normalization.json
#[derive(Debug, Clone, Deserialize)]
pub struct NormalizationConstants {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

/// A reference vector from the dataset
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceVector {
    pub vector: [f32; 14],
    pub label: String,
}

/// MCC risk mapping
pub type MccRiskMap = HashMap<String, f32>;

/// Load normalization constants from JSON file
pub fn load_normalization(path: &str) -> Result<NormalizationConstants, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read normalization file: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse normalization: {}", e))
}

/// Load MCC risk map from JSON file
pub fn load_mcc_risk(path: &str) -> Result<MccRiskMap, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read mcc_risk file: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse mcc_risk: {}", e))
}
/// Parse references from a JSON array using streaming deserialization.
fn parse_json_array_stream<R: std::io::Read>(reader: R) -> Result<(Vec<u16>, Vec<u8>), String> {
    use std::io::BufReader;
    let buf = BufReader::with_capacity(1 << 20, reader);
    // serde_json::from_reader with Vec<ReferenceVector> uses incremental parsing,
    // never holding the full AST in memory (only the deserialized structs).
    let refs: Vec<ReferenceVector> =
        serde_json::from_reader(buf).map_err(|e| format!("Failed to parse references: {e}"))?;

    let mut vectors = Vec::with_capacity(refs.len() * 14);
    let mut labels = Vec::with_capacity(refs.len());

    for entry in &refs {
        for &val in &entry.vector {
            let f16_val = half::f16::from_f32(val);
            vectors.push(f16_val.to_bits());
        }
        labels.push(match entry.label.as_str() {
            "fraud" => 1u8,
            _ => 0u8,
        });
    }

    Ok((vectors, labels))
}

/// Parse references from a JSON array (non-gzipped, small files)
#[allow(dead_code)]
fn parse_json_array<R: std::io::Read>(reader: R) -> Result<(Vec<u16>, Vec<u8>), String> {
    let refs: Vec<ReferenceVector> = serde_json::from_reader(reader)
        .map_err(|e| format!("Failed to parse references: {}", e))?;

    let mut vectors = Vec::with_capacity(refs.len() * 14);
    let mut labels = Vec::with_capacity(refs.len());

    for entry in &refs {
        for &val in &entry.vector {
            let f16_val = half::f16::from_f32(val);
            vectors.push(f16_val.to_bits());
        }
        labels.push(match entry.label.as_str() {
            "fraud" => 1u8,
            _ => 0u8,
        });
    }

    Ok((vectors, labels))
}

/// Load references from JSON array, supports both `.gz` and plain JSON
pub fn load_references(path: &str) -> Result<(Vec<u16>, Vec<u8>), String> {
    let file =
        fs::File::open(path).map_err(|e| format!("Failed to open references file: {}", e))?;

    if path.ends_with(".gz") {
        let decoder = GzDecoder::new(file);
        // Use streaming parser for large files
        parse_json_array_stream(decoder)
    } else {
        parse_json_array(file)
    }
}
