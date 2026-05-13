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
pub struct Payload {
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    #[serde(default)]
    pub last_transaction: Option<LastTransaction>,
}

pub fn parse(buf: &[u8]) -> Option<VecPayload> {
    let p: Payload = serde_json::from_slice(buf).ok()?;
    let req = &p.transaction.requested_at;
    let (y, mo, d, h, min) = parse_iso_str(req)?;
    let hour = h as u8;
    let day_of_week = day_of_week(y, mo, d) as u8;

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
        day_of_week,
        is_online: p.terminal.is_online,
        card_present: p.terminal.card_present,
        is_unknown_merchant,
        has_last_tx,
    })
}

fn parse_iso_str(s: &str) -> Option<(u32, u32, u32, u32, u32)> {
    let b = s.as_bytes();
    if b.len() < 16 {
        return None;
    }
    let y = parse4(b, 0);
    let mo = parse2(b, 5);
    let d = parse2(b, 8);
    let h = parse2(b, 11);
    let min = parse2(b, 14);
    Some((y, mo, d, h, min))
}

fn parse4(s: &[u8], off: usize) -> u32 {
    (s[off] - b'0') as u32 * 1000
        + (s[off + 1] - b'0') as u32 * 100
        + (s[off + 2] - b'0') as u32 * 10
        + (s[off + 3] - b'0') as u32
}

fn parse2(s: &[u8], off: usize) -> u32 {
    (s[off] - b'0') as u32 * 10 + (s[off + 1] - b'0') as u32
}

fn days_since_epoch(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = (y - era * 400) as u32;
    let mm = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mm + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146097 + doe as i64 - 719468
}

fn minutes_between(
    y1: u32, mo1: u32, d1: u32, h1: u32, min1: u32,
    y2: u32, mo2: u32, d2: u32, h2: u32, min2: u32,
) -> u32 {
    let d1 = days_since_epoch(y1 as i32, mo1, d1);
    let d2 = days_since_epoch(y2 as i32, mo2, d2);
    let m1 = d1 * 1440 + h1 as i64 * 60 + min1 as i64;
    let m2 = d2 * 1440 + h2 as i64 * 60 + min2 as i64;
    (m2 - m1).max(0) as u32
}

fn day_of_week(y: u32, m: u32, d: u32) -> u8 {
    let t = [0u32, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let ya = if m < 3 { y - 1 } else { y };
    let dow = (ya + ya / 4 - ya / 100 + ya / 400 + t[(m - 1) as usize] + d) % 7;
    ((dow + 6) % 7) as u8
}
