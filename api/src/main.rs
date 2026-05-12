use axum::body::Bytes;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Router,
};
use rinha_core::vector::VectorSearch;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Fast MCC lookup without HashMap
fn mcc_risk(mcc: &[u8]) -> f32 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.50,
    }
}

/// Normalization constants (inline)
const MAX_AMOUNT: f32 = 10000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT: f32 = 20.0;
const MAX_MERCHANT_AVG: f32 = 10000.0;

/// Pre-baked responses: index = (score * 10) as usize / 2
static PREBAKED: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

struct AppState {
    search: Arc<VectorSearch>,
}

type SharedState = Arc<AppState>;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("rinha_api=info")
        .init();
    let start = Instant::now();

    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".to_string());
    let index_path = std::env::var("INDEX_PATH")
        .unwrap_or_else(|_| format!("{}/index.bin.gz", data_dir));

    info!("Loading index from {}...", index_path);
    let search = rinha_core::vector::VectorSearch::load(&index_path);

    info!(
        "Index loaded in {:?}. ~{} MB, {} centroids, {} blocks, {} vectors",
        start.elapsed(),
        search.memory_usage() / 1048576,
        search.k,
        search.total_blocks,
        search.count,
    );

    info!("Startup complete: {:?}", start.elapsed());

    let state = Arc::new(AppState { search: Arc::new(search) });
    let app = Router::new()
        .route("/ready", get(|| async { StatusCode::OK }))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    info!("Listening on 0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn fraud_score(
    State(state): State<SharedState>,
    body: Bytes,
) -> Result<&'static [u8], StatusCode> {
    let q = parse_body_fast(&body)?;

    // Single-threaded runtime: inline the search (no spawn_blocking overhead)
    let neighbors = state.search.search_with_probe(&q, 2);

    let fraud_count = neighbors.iter().filter(|r| r.label == 1).count();
    let n = neighbors.len().max(1);
    let score = fraud_count as f32 / n as f32;
    let idx = ((score * 10.0).round() as usize).min(10) / 2;
    Ok(PREBAKED[idx])
}

// ---------------------------------------------------------------------------
// Minimal JSON parser — extracts fields needed for the 14D vector.
// Zero allocations, one pass over the input.
// ---------------------------------------------------------------------------

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
    tx_amount: f32,
    tx_installments: i32,
    tx_requested_at: &'a [u8],
    cust_avg_amount: f32,
    cust_tx_count: i32,
    known_merchants: [&'a [u8]; 20],
    known_count: usize,
    merch_id: &'a [u8],
    merch_mcc: &'a [u8],
    merch_avg_amount: f32,
    term_online: bool,
    term_card_present: bool,
    term_km: f32,
    last_ts: &'a [u8],
    last_km: f32,
    has_last: bool,
}

impl<'a> Parser<'a> {
    fn new(s: &'a [u8]) -> Self {
        Self {
            s,
            pos: 0,
            tx_amount: 0.0,
            tx_installments: 0,
            tx_requested_at: b"",
            cust_avg_amount: 0.0,
            cust_tx_count: 0,
            known_merchants: [b""; 20],
            known_count: 0,
            merch_id: b"",
            merch_mcc: b"",
            merch_avg_amount: 0.0,
            term_online: false,
            term_card_present: false,
            term_km: 0.0,
            last_ts: b"",
            last_km: 0.0,
            has_last: false,
        }
    }

    fn peek(&self) -> u8 {
        if self.pos < self.s.len() {
            self.s[self.pos]
        } else {
            0
        }
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn skip_ws(&mut self) {
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) -> bool {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn read_string(&mut self) -> &'a [u8] {
        self.skip_ws();
        if self.pos >= self.s.len() || self.s[self.pos] != b'"' {
            return b"";
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.s.len() && self.s[self.pos] != b'"' {
            self.pos += 1;
        }
        let result = &self.s[start..self.pos];
        if self.pos < self.s.len() {
            self.pos += 1;
        }
        result
    }

    fn read_number(&mut self) -> f32 {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.s.len()
            && (self.s[self.pos].is_ascii_digit()
                || self.s[self.pos] == b'.'
                || self.s[self.pos] == b'-'
                || self.s[self.pos] == b'+'
                || self.s[self.pos] == b'e'
                || self.s[self.pos] == b'E')
        {
            self.pos += 1;
        }
        if self.pos > start {
            let slice = unsafe { std::str::from_utf8_unchecked(&self.s[start..self.pos]) };
            slice.parse::<f32>().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    fn read_int(&mut self) -> i32 {
        self.read_number() as i32
    }

    fn read_bool(&mut self) -> bool {
        self.skip_ws();
        if self.pos + 4 <= self.s.len() && &self.s[self.pos..self.pos + 4] == b"true" {
            self.pos += 4;
            true
        } else if self.pos + 5 <= self.s.len() && &self.s[self.pos..self.pos + 5] == b"false" {
            self.pos += 5;
            false
        } else {
            false
        }
    }

    fn read_null(&mut self) {
        self.skip_ws();
        if self.pos + 4 <= self.s.len() && &self.s[self.pos..self.pos + 4] == b"null" {
            self.pos += 4;
        }
    }

    fn skip_value(&mut self) {
        self.skip_ws();
        if self.pos >= self.s.len() {
            return;
        }
        match self.s[self.pos] {
            b'"' => {
                self.read_string();
            }
            b't' | b'f' => {
                self.read_bool();
            }
            b'n' => {
                self.read_null();
            }
            b'[' => {
                self.pos += 1;
                let mut depth = 1;
                while self.pos < self.s.len() && depth > 0 {
                    match self.s[self.pos] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        b'"' => {
                            self.read_string();
                            continue;
                        }
                        _ => {}
                    }
                    self.pos += 1;
                }
            }
            b'{' => {
                self.pos += 1;
                let mut depth = 1;
                while self.pos < self.s.len() && depth > 0 {
                    match self.s[self.pos] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        b'"' => {
                            self.read_string();
                            continue;
                        }
                        _ => {}
                    }
                    self.pos += 1;
                }
            }
            _ => {
                while self.pos < self.s.len()
                    && (self.s[self.pos].is_ascii_digit()
                        || self.s[self.pos] == b'.'
                        || self.s[self.pos] == b'-'
                        || self.s[self.pos] == b'+'
                        || self.s[self.pos] == b'e'
                        || self.s[self.pos] == b'E')
                {
                    self.pos += 1;
                }
            }
        }
    }

    fn read_key(&mut self) -> &'a [u8] {
        self.read_string()
    }

    fn expect_colon(&mut self) {
        self.skip_ws();
        if self.pos < self.s.len() && self.s[self.pos] == b':' {
            self.pos += 1;
        }
        self.skip_ws();
    }

    fn parse_all(&mut self) {
        if !self.expect(b'{') {
            return;
        }
        while self.pos < self.s.len() {
            self.skip_ws();
            match self.peek() {
                b'}' => {
                    self.advance();
                    break;
                }
                b'"' => {
                    let key = self.read_key();
                    self.expect_colon();
                    match key {
                        b"id" => self.skip_value(),
                        b"transaction" => self.parse_tx(),
                        b"customer" => self.parse_cust(),
                        b"merchant" => self.parse_merch(),
                        b"terminal" => self.parse_term(),
                        b"last_transaction" => {
                            self.skip_ws();
                            if self.peek() == b'n' {
                                self.read_null();
                                self.has_last = false;
                            } else {
                                self.has_last = true;
                                self.parse_last();
                            }
                        }
                        _ => self.skip_value(),
                    }
                }
                b',' => { self.advance(); }
                _ => { self.advance(); }
            }
        }
    }

    fn parse_tx(&mut self) {
        if !self.expect(b'{') { return; }
        while self.pos < self.s.len() {
            self.skip_ws();
            if self.peek() == b'}' { self.advance(); break; }
            let key = self.read_key();
            self.expect_colon();
            match key {
                b"amount" => self.tx_amount = self.read_number(),
                b"installments" => self.tx_installments = self.read_int(),
                b"requested_at" => self.tx_requested_at = self.read_string(),
                _ => self.skip_value(),
            }
            self.expect(b',');
        }
    }

    fn parse_cust(&mut self) {
        if !self.expect(b'{') { return; }
        while self.pos < self.s.len() {
            self.skip_ws();
            if self.peek() == b'}' { self.advance(); break; }
            let key = self.read_key();
            self.expect_colon();
            match key {
                b"avg_amount" => self.cust_avg_amount = self.read_number(),
                b"tx_count_24h" => self.cust_tx_count = self.read_int(),
                b"known_merchants" => {
                    if self.expect(b'[') {
                        self.known_count = 0;
                        while self.pos < self.s.len()
                            && self.peek() != b']'
                            && self.known_count < 20
                        {
                            let m = self.read_string();
                            if !m.is_empty() {
                                self.known_merchants[self.known_count] = m;
                                self.known_count += 1;
                            }
                            self.expect(b',');
                        }
                        self.expect(b']');
                    }
                }
                _ => self.skip_value(),
            }
            self.expect(b',');
        }
    }

    fn parse_merch(&mut self) {
        if !self.expect(b'{') { return; }
        while self.pos < self.s.len() {
            self.skip_ws();
            if self.peek() == b'}' { self.advance(); break; }
            let key = self.read_key();
            self.expect_colon();
            match key {
                b"id" => self.merch_id = self.read_string(),
                b"mcc" => self.merch_mcc = self.read_string(),
                b"avg_amount" => self.merch_avg_amount = self.read_number(),
                _ => self.skip_value(),
            }
            self.expect(b',');
        }
    }

    fn parse_term(&mut self) {
        if !self.expect(b'{') { return; }
        while self.pos < self.s.len() {
            self.skip_ws();
            if self.peek() == b'}' { self.advance(); break; }
            let key = self.read_key();
            self.expect_colon();
            match key {
                b"is_online" => self.term_online = self.read_bool(),
                b"card_present" => self.term_card_present = self.read_bool(),
                b"km_from_home" => self.term_km = self.read_number(),
                _ => self.skip_value(),
            }
            self.expect(b',');
        }
    }

    fn parse_last(&mut self) {
        if !self.expect(b'{') { return; }
        while self.pos < self.s.len() {
            self.skip_ws();
            if self.peek() == b'}' { self.advance(); break; }
            let key = self.read_key();
            self.expect_colon();
            match key {
                b"timestamp" => self.last_ts = self.read_string(),
                b"km_from_current" => self.last_km = self.read_number(),
                _ => self.skip_value(),
            }
            self.expect(b',');
        }
    }

    fn build_vector(&self) -> [f32; 14] {
        let mut q = [0.0f32; 14];
        q[0] = (self.tx_amount / MAX_AMOUNT).clamp(0.0, 1.0);
        q[1] = (self.tx_installments as f32 / MAX_INSTALLMENTS).clamp(0.0, 1.0);
        q[2] = ((self.tx_amount / self.cust_avg_amount) / AMOUNT_VS_AVG_RATIO).clamp(0.0, 1.0);
        if !self.tx_requested_at.is_empty() {
            q[3] = parse_hour_raw(self.tx_requested_at) as f32 / 23.0;
            q[4] = parse_dow_raw(self.tx_requested_at) as f32 / 6.0;
        }
        if self.has_last && !self.last_ts.is_empty() {
            let minutes = minutes_between_raw(self.last_ts, self.tx_requested_at).max(0) as f32;
            q[5] = (minutes / MAX_MINUTES).clamp(0.0, 1.0);
            q[6] = (self.last_km / MAX_KM).clamp(0.0, 1.0);
        } else {
            q[5] = -1.0;
            q[6] = -1.0;
        }
        q[7] = (self.term_km / MAX_KM).clamp(0.0, 1.0);
        q[8] = (self.cust_tx_count as f32 / MAX_TX_COUNT).clamp(0.0, 1.0);
        q[9] = if self.term_online { 1.0 } else { 0.0 };
        q[10] = if self.term_card_present { 1.0 } else { 0.0 };
        q[11] = if self.known_merchants[..self.known_count]
            .iter()
            .any(|m| *m == self.merch_id)
        { 0.0 } else { 1.0 };
        q[12] = mcc_risk(self.merch_mcc);
        q[13] = (self.merch_avg_amount / MAX_MERCHANT_AVG).clamp(0.0, 1.0);
        q
    }
}

fn parse_body_fast(body: &[u8]) -> Result<[f32; 14], StatusCode> {
    let mut p = Parser::new(body);
    p.parse_all();
    Ok(p.build_vector())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_legit_example() {
        let body = br#"{"id":"tx-1329056812","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}"#;
        let q = parse_body_fast(body).unwrap();
        eprintln!("LEGIT: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13]);
        assert!((q[0] - 0.0041).abs() < 0.001, "dim0: {}", q[0]);
        assert!((q[1] - 0.1667).abs() < 0.001, "dim1: {}", q[1]);
        assert!((q[2] - 0.05).abs() < 0.001, "dim2: {}", q[2]);
        assert!((q[5] + 1.0).abs() < 0.001, "dim5: {}", q[5]);
        assert!((q[6] + 1.0).abs() < 0.001, "dim6: {}", q[6]);
        assert!((q[7] - 0.0292).abs() < 0.001, "dim7: {}", q[7]);
        assert!((q[8] - 0.15).abs() < 0.001, "dim8: {}", q[8]);
        assert!((q[12] - 0.15).abs() < 0.001, "dim12: {}", q[12]);
    }

    #[test]
    fn test_parser_fraud_example() {
        let body = br#"{"id":"tx-3330991687","transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},"customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},"merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},"terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},"last_transaction":null}"#;
        let q = parse_body_fast(body).unwrap();
        eprintln!("FRAUD: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13]);
        assert!((q[0] - 0.9506).abs() < 0.001, "dim0: {}", q[0]);
        assert!((q[1] - 0.8333).abs() < 0.001, "dim1: {}", q[1]);
        assert!((q[2] - 1.0).abs() < 0.001, "dim2: {}", q[2]);
        assert!((q[7] - 0.9523).abs() < 0.001, "dim7: {}", q[7]);
        assert!((q[8] - 1.0).abs() < 0.001, "dim8: {}", q[8]);
        assert!((q[12] - 0.75).abs() < 0.001, "dim12: {}", q[12]);
    }
}

fn parse_hour_raw(iso: &[u8]) -> u8 {
    if iso.len() < 13 { return 0; }
    (iso[11] - b'0') * 10 + (iso[12] - b'0')
}

fn parse_dow_raw(iso: &[u8]) -> u8 {
    if iso.len() < 10 { return 0; }
    let y = (iso[0] - b'0') as i32 * 1000 + (iso[1] - b'0') as i32 * 100
        + (iso[2] - b'0') as i32 * 10 + (iso[3] - b'0') as i32;
    let m = (iso[5] - b'0') as u32 * 10 + (iso[6] - b'0') as u32;
    let d = (iso[8] - b'0') as u32 * 10 + (iso[9] - b'0') as u32;
    let (ym, mm) = if m < 3 { (y - 1, m + 12) } else { (y, m) };
    static T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    ((ym as u32 + ym as u32 / 4 - ym as u32 / 100 + ym as u32 / 400 + T[mm as usize - 1] + d + 6) % 7) as u8
}

fn minutes_between_raw(start: &[u8], end: &[u8]) -> i64 {
    fn parse_minutes(iso: &[u8]) -> i64 {
        if iso.len() < 19 { return 0; }
        let y = (iso[0] - b'0') as i64 * 1000 + (iso[1] - b'0') as i64 * 100
            + (iso[2] - b'0') as i64 * 10 + (iso[3] - b'0') as i64;
        let mo = (iso[5] - b'0') as i64 * 10 + (iso[6] - b'0') as i64;
        let d = (iso[8] - b'0') as i64 * 10 + (iso[9] - b'0') as i64;
        let h = (iso[11] - b'0') as i64 * 10 + (iso[12] - b'0') as i64;
        let mi = (iso[14] - b'0') as i64 * 10 + (iso[15] - b'0') as i64;
        let s = (iso[17] - b'0') as i64 * 10 + (iso[18] - b'0') as i64;
        let (ym, mm) = if mo < 3 { (y - 1, mo + 12) } else { (y, mo) };
        let days = ym * 365 + ym / 4 - ym / 100 + ym / 400 + (mm * 306 + 5) / 10 + d - 1;
        days * 24 * 60 + h * 60 + mi + s / 60
    }
    (parse_minutes(end) - parse_minutes(start)).max(0)
}
