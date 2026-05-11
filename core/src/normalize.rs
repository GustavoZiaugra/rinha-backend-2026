use crate::dataset::NormalizationConstants;
use std::collections::HashMap;

/// The complete normalized 14D vector for fraud detection
pub type Vector14D = [f32; 14];

/// Incoming transaction payload structure
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TransactionPayload {
    pub id: String,
    pub transaction: TransactionData,
    pub customer: CustomerData,
    pub merchant: MerchantData,
    pub terminal: TerminalData,
    pub last_transaction: Option<LastTransactionData>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TransactionData {
    pub amount: f32,
    pub installments: i32,
    pub requested_at: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CustomerData {
    pub avg_amount: f32,
    pub tx_count_24h: i32,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MerchantData {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TerminalData {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LastTransactionData {
    pub timestamp: String,
    pub km_from_current: f32,
}

/// Normalize a transaction payload into a 14D vector
///
/// Formulas from REGRAS_DE_DETECCAO.md
pub fn normalize_transaction(
    payload: &TransactionPayload,
    norm: &NormalizationConstants,
    mcc_risk: &HashMap<String, f32>,
) -> Vector14D {
    let mut v = [0.0f32; 14];

    // Index 0: amount / max_amount
    v[0] = clamp(payload.transaction.amount / norm.max_amount);

    // Index 1: installments / max_installments
    v[1] = clamp(payload.transaction.installments as f32 / norm.max_installments);

    // Index 2: (amount / customer.avg_amount) / amount_vs_avg_ratio
    v[2] = clamp(
        (payload.transaction.amount / payload.customer.avg_amount) / norm.amount_vs_avg_ratio,
    );

    // Index 3: hour_of_day / 23 (UTC)
    let hour = extract_hour(&payload.transaction.requested_at);
    v[3] = hour as f32 / 23.0;

    // Index 4: day_of_week / 6 (mon=0, sun=6)
    let dow = extract_day_of_week(&payload.transaction.requested_at);
    v[4] = dow as f32 / 6.0;

    // Index 5: minutes_since_last_tx
    // Index 6: km_from_last_tx
    if let Some(ref last_tx) = payload.last_transaction {
        let minutes = minutes_between(&last_tx.timestamp, &payload.transaction.requested_at);
        v[5] = clamp(minutes as f32 / norm.max_minutes);
        v[6] = clamp(last_tx.km_from_current / norm.max_km);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    // Index 7: km_from_home / max_km
    v[7] = clamp(payload.terminal.km_from_home / norm.max_km);

    // Index 8: tx_count_24h / max_tx_count_24h
    v[8] = clamp(payload.customer.tx_count_24h as f32 / norm.max_tx_count_24h);

    // Index 9: is_online
    v[9] = if payload.terminal.is_online { 1.0 } else { 0.0 };

    // Index 10: card_present
    v[10] = if payload.terminal.card_present {
        1.0
    } else {
        0.0
    };

    // Index 11: unknown_merchant
    let is_known = payload
        .customer
        .known_merchants
        .iter()
        .any(|m| m == &payload.merchant.id);
    v[11] = if is_known { 0.0 } else { 1.0 };

    // Index 12: mcc_risk
    v[12] = *mcc_risk.get(&payload.merchant.mcc).unwrap_or(&0.5);

    // Index 13: merchant_avg_amount / max_merchant_avg_amount
    v[13] = clamp(payload.merchant.avg_amount / norm.max_merchant_avg_amount);

    v
}

/// Clamp value to [0.0, 1.0]
fn clamp(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Extract hour (0-23) from ISO 8601 timestamp
fn extract_hour(iso: &str) -> u8 {
    // Format: "2026-03-11T20:23:35Z"
    if iso.len() >= 16 {
        if let Ok(h) = iso[11..13].parse::<u8>() {
            return h;
        }
    }
    0
}

/// Extract day of week from ISO 8601 timestamp
/// Using a simple approach: parse date and compute day of week
fn extract_day_of_week(iso: &str) -> u8 {
    // Format: "2026-03-11T20:23:35Z"
    if iso.len() >= 10 {
        if let (Ok(year), Ok(month), Ok(day)) = (
            iso[0..4].parse::<i32>(),
            iso[5..7].parse::<u32>(),
            iso[8..10].parse::<u32>(),
        ) {
            // Tomohiko Sakamoto's algorithm for day of week
            // mon=0, sun=6
            return day_of_week(year, month, day);
        }
    }
    0
}

/// Tomohiko Sakamoto's day-of-week algorithm
/// Returns 0=Monday ... 6=Sunday
fn day_of_week(y: i32, m: u32, d: u32) -> u8 {
    static T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let year = if m < 3 { y - 1 } else { y };
    ((year as u32 + year as u32 / 4 - year as u32 / 100
        + year as u32 / 400
        + T[(m - 1) as usize]
        + d
        + 6)
        % 7) as u8
}

/// Calculate minutes between two ISO 8601 timestamps
fn minutes_between(start: &str, end: &str) -> i64 {
    // Simple parse: extract YYYY-MM-DDTHH:MM:SS
    fn parse_minutes(iso: &str) -> i64 {
        if iso.len() < 19 {
            return 0;
        }
        let y = iso[0..4].parse::<i64>().unwrap_or(0);
        let mo = iso[5..7].parse::<i64>().unwrap_or(0);
        let d = iso[8..10].parse::<i64>().unwrap_or(0);
        let h = iso[11..13].parse::<i64>().unwrap_or(0);
        let mi = iso[14..16].parse::<i64>().unwrap_or(0);
        let s = iso[17..19].parse::<i64>().unwrap_or(0);
        // Convert to minutes since epoch (simplified)
        days_since_epoch(y, mo, d) * 24 * 60 + h * 60 + mi + s / 60
    }

    let start_min = parse_minutes(start);
    let end_min = parse_minutes(end);
    (end_min - start_min).max(0)
}

fn days_since_epoch(y: i64, m: i64, d: i64) -> i64 {
    let m = (m + 9) % 12;
    let y = y - m / 10;
    y * 365 + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + d - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_day_of_week() {
        // 2026-03-11 is a Wednesday = 2 (mon=0)
        assert_eq!(extract_day_of_week("2026-03-11T20:23:35Z"), 2);
    }

    #[test]
    fn test_hour() {
        assert_eq!(extract_hour("2026-03-11T20:23:35Z"), 20);
        assert_eq!(extract_hour("2026-03-11T05:15:12Z"), 5);
    }

    #[test]
    fn test_normalize_example() {
        // Test with the legit example from the docs
        let norm = NormalizationConstants {
            max_amount: 10000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10000.0,
        };

        let mut mcc_risk = HashMap::new();
        mcc_risk.insert("5411".to_string(), 0.15);
        mcc_risk.insert("7802".to_string(), 0.75);

        let payload = TransactionPayload {
            id: "tx-1329056812".to_string(),
            transaction: TransactionData {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".to_string(),
            },
            customer: CustomerData {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".to_string(), "MERC-016".to_string()],
            },
            merchant: MerchantData {
                id: "MERC-016".to_string(),
                mcc: "5411".to_string(),
                avg_amount: 60.25,
            },
            terminal: TerminalData {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        };

        let v = normalize_transaction(&payload, &norm, &mcc_risk);

        // Expected from docs: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
        assert!((v[0] - 0.0041).abs() < 0.001, "dim0: {}", v[0]);
        assert!((v[1] - 0.1667).abs() < 0.001, "dim1: {}", v[1]);
        assert!((v[2] - 0.05).abs() < 0.001, "dim2: {}", v[2]);
        assert!((v[3] - 0.7826).abs() < 0.001, "dim3: {}", v[3]);
        assert!((v[4] - 0.3333).abs() < 0.001, "dim4: {}", v[4]);
        assert!((v[5] + 1.0).abs() < 0.001, "dim5: {}", v[5]); // -1 sentinel
        assert!((v[6] + 1.0).abs() < 0.001, "dim6: {}", v[6]); // -1 sentinel
        assert!((v[7] - 0.0292).abs() < 0.001, "dim7: {}", v[7]);
        assert!((v[8] - 0.15).abs() < 0.001, "dim8: {}", v[8]);
        assert!((v[9] - 0.0).abs() < 0.001, "dim9: {}", v[9]);
        assert!((v[10] - 1.0).abs() < 0.001, "dim10: {}", v[10]);
        assert!((v[11] - 0.0).abs() < 0.001, "dim11: {}", v[11]);
        assert!((v[12] - 0.15).abs() < 0.001, "dim12: {}", v[12]);
        assert!((v[13] - 0.006).abs() < 0.001, "dim13: {}", v[13]);
    }

    #[test]
    fn test_normalize_fraud_example() {
        let norm = NormalizationConstants {
            max_amount: 10000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10000.0,
        };

        let mut mcc_risk = HashMap::new();
        mcc_risk.insert("7802".to_string(), 0.75);

        let payload = TransactionPayload {
            id: "tx-3330991687".to_string(),
            transaction: TransactionData {
                amount: 9505.97,
                installments: 10,
                requested_at: "2026-03-14T05:15:12Z".to_string(),
            },
            customer: CustomerData {
                avg_amount: 81.28,
                tx_count_24h: 20,
                known_merchants: vec![
                    "MERC-008".to_string(),
                    "MERC-007".to_string(),
                    "MERC-005".to_string(),
                ],
            },
            merchant: MerchantData {
                id: "MERC-068".to_string(),
                mcc: "7802".to_string(),
                avg_amount: 54.86,
            },
            terminal: TerminalData {
                is_online: false,
                card_present: true,
                km_from_home: 952.27,
            },
            last_transaction: None,
        };

        let v = normalize_transaction(&payload, &norm, &mcc_risk);

        // Expected: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
        assert!((v[0] - 0.9506).abs() < 0.001, "dim0: {}", v[0]);
        assert!((v[1] - 0.8333).abs() < 0.001, "dim1: {}", v[1]);
        assert!((v[2] - 1.0).abs() < 0.001, "dim2: {}", v[2]);
        assert!((v[3] - 0.2174).abs() < 0.001, "dim3: {}", v[3]);
        assert!((v[4] - 0.8333).abs() < 0.001, "dim4: {}", v[4]);
        assert!((v[5] + 1.0).abs() < 0.001, "dim5: {}", v[5]);
        assert!((v[6] + 1.0).abs() < 0.001, "dim6: {}", v[6]);
        assert!((v[7] - 0.9523).abs() < 0.001, "dim7: {}", v[7]);
        assert!((v[8] - 1.0).abs() < 0.001, "dim8: {}", v[8]);
        assert!((v[9] - 0.0).abs() < 0.001, "dim9: {}", v[9]);
        assert!((v[10] - 1.0).abs() < 0.001, "dim10: {}", v[10]);
        assert!((v[11] - 1.0).abs() < 0.001, "dim11: {}", v[11]);
        assert!((v[12] - 0.75).abs() < 0.001, "dim12: {}", v[12]);
        assert!((v[13] - 0.0055).abs() < 0.001, "dim13: {}", v[13]);
    }
}
