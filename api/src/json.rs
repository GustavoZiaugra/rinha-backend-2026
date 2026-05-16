use serde::Deserialize;
use crate::vector::Payload as VecPayload;

#[derive(Debug, Deserialize)]
pub struct Transaction {
    pub amount: f32,
    pub installments: i32,
    pub requested_at: String,
}

#[derive(Debug, Deserialize)]
pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: i32,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Debug, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f32,
}

#[derive(Debug, Deserialize)]
struct RawPayload {
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    #[serde(default)]
    pub last_transaction: Option<LastTransaction>,
}

pub fn parse(buf: &[u8]) -> Option<VecPayload> {
    let p: RawPayload = serde_json::from_slice(buf).ok()?;
    let req = &p.transaction.requested_at;
    let (y, mo, d, h, min) = parse_iso_str(req)?;
    let hour = h as u8;

    let (has_last_tx, minutes_since_last, km_from_current) = match &p.last_transaction {
        Some(lt) => {
            let (y2, mo2, d2, h2, min2) = parse_iso_str(&lt.timestamp)?;
            let mins = minutes_between(y2, mo2, d2, h2, min2, y, mo, d, h, min);
            (true, mins, lt.km_from_current)
        }
        None => (false, 0, 0.0),
    };

    let is_unknown_merchant = !p.customer.known_merchants.iter().any(|s| *s == p.merchant.id);
    let mcc = p.merchant.mcc.parse().unwrap_or(0);

    Some(VecPayload {
        amount: p.transaction.amount,
        customer_avg_amount: p.customer.avg_amount,
        merchant_avg_amount: p.merchant.avg_amount,
        km_from_home: p.terminal.km_from_home,
        km_from_current,
        tx_count_24h: p.customer.tx_count_24h as u32,
        mcc,
        minutes_since_last,
        installments: p.transaction.installments as u8,
        hour,
        day_of_week: day_of_week(y, mo, d) as u8,
        is_online: p.terminal.is_online,
        card_present: p.terminal.card_present,
        is_unknown_merchant,
        has_last_tx,
    })
}

#[inline(always)]
fn parse_digit(b: u8) -> Option<u32> {
    if b.is_ascii_digit() { Some((b - b'0') as u32) } else { None }
}

#[inline(always)]
fn parse_iso_str(s: &str) -> Option<(u32, u32, u32, u32, u32)> {
    let b = s.as_bytes();
    if b.len() < 16 { return None; }
    let y = parse_digit(b[0])? * 1000 + parse_digit(b[1])? * 100 + parse_digit(b[2])? * 10 + parse_digit(b[3])?;
    let mo = parse_digit(b[5])? * 10 + parse_digit(b[6])?;
    let d = parse_digit(b[8])? * 10 + parse_digit(b[9])?;
    let h = parse_digit(b[11])? * 10 + parse_digit(b[12])?;
    let min = parse_digit(b[14])? * 10 + parse_digit(b[15])?;
    if mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || min > 59 { return None; }
    Some((y, mo, d, h, min))
}

fn minutes_between(y1: u32, mo1: u32, d1: u32, h1: u32, m1: u32, y2: u32, mo2: u32, d2: u32, h2: u32, m2: u32) -> u32 {
    let t1 = y1 as i64 * 525600 + mo1 as i64 * 43200 + d1 as i64 * 1440 + h1 as i64 * 60 + m1 as i64;
    let t2 = y2 as i64 * 525600 + mo2 as i64 * 43200 + d2 as i64 * 1440 + h2 as i64 * 60 + m2 as i64;
    let diff = t2 - t1;
    if diff > 0 { diff as u32 } else { 0 }
}

fn day_of_week(y: u32, m: u32, d: u32) -> u32 {
    // Tomohiko Sakamoto's algorithm
    let t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    (y + y / 4 - y / 100 + y / 400 + t[(m - 1) as usize] + d) % 7
}
