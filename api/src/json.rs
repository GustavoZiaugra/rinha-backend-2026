use crate::vector::Payload as VecPayload;

/// Manual byte-level JSON parser — no heap allocations, no str::parse.
/// Uses memchr2 for fast field scanning and manual f32/u32 parsers.

/// Power-of-10 table for fast fractional conversion
static FRAC_POWERS: [f64; 19] = [
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 
    1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18,
];

pub fn parse(buf: &[u8]) -> Option<VecPayload> {
    let mut p = 0usize;

    to_next_value(&mut p, buf)?;
    skip_string(&mut p, buf)?;

    to_next_value(&mut p, buf)?;
    to_next_value(&mut p, buf)?;
    let amount = scan_f32(&mut p, buf);

    to_next_value(&mut p, buf)?;
    let installments = scan_u32(&mut p, buf) as u8;

    to_next_value(&mut p, buf)?;
    let (req_y, req_mo, req_d, req_h, req_min) = scan_iso(&mut p, buf)?;

    to_next_value(&mut p, buf)?;
    to_next_value(&mut p, buf)?;
    let customer_avg_amount = scan_f32(&mut p, buf);

    to_next_value(&mut p, buf)?;
    let tx_count_24h = scan_u32(&mut p, buf);

    to_next_value(&mut p, buf)?;
    // known_merchants array — collect up to 16 slices
    p += 1; // skip [
    let mut merchant_slices: [&[u8]; 16] = [&[]; 16];
    let mut mc = 0usize;
    while p < buf.len() && buf[p] != b']' {
        if buf[p] == b'"' {
            p += 1;
            let start = p;
            let end = memchr::memchr(b'"', &buf[p..])?;
            if mc < 16 {
                merchant_slices[mc] = &buf[start..start + end];
                mc += 1;
            }
            p = start + end + 1;
        } else {
            p += 1;
        }
    }
    if p < buf.len() {
        p += 1; // skip ]
    }

    to_next_value(&mut p, buf)?;
    to_next_value(&mut p, buf)?;
    let merchant_id = scan_string(&mut p, buf)?;

    to_next_value(&mut p, buf)?;
    let mcc = scan_mcc(&mut p, buf);

    to_next_value(&mut p, buf)?;
    let merchant_avg_amount = scan_f32(&mut p, buf);

    to_next_value(&mut p, buf)?;
    to_next_value(&mut p, buf)?;
    let is_online = scan_bool(&mut p, buf);

    to_next_value(&mut p, buf)?;
    let card_present = scan_bool(&mut p, buf);

    to_next_value(&mut p, buf)?;
    let km_from_home = scan_f32(&mut p, buf);

    // last_transaction — may be null
    to_next_value(&mut p, buf)?;
    let has_last_tx = p < buf.len() && buf[p] != b'n';
    let (minutes_since_last, km_from_current) = if has_last_tx {
        to_next_value(&mut p, buf)?;
        let (ly, lmo, ld, lh, lmin) = scan_iso(&mut p, buf)?;
        to_next_value(&mut p, buf)?;
        let km = scan_f32(&mut p, buf);
        let mins = minutes_between(ly, lmo, ld, lh, lmin, req_y, req_mo, req_d, req_h, req_min);
        (mins, km)
    } else {
        (0, 0.0)
    };

    let is_unknown_merchant = !merchant_slices[..mc].iter().any(|&m| m == merchant_id);

    Some(VecPayload {
        amount,
        installments,
        hour: req_h,
        day_of_week: day_of_week(req_y, req_mo, req_d),
        customer_avg_amount,
        tx_count_24h,
        mcc,
        merchant_avg_amount,
        is_online,
        card_present,
        km_from_home,
        is_unknown_merchant,
        has_last_tx,
        minutes_since_last,
        km_from_current,
    })
}

// ---------- Navigation helpers ----------

/// Advance to the next `:value` pair by scanning for `:` or `"` (skip strings)
#[inline]
fn to_next_value(p: &mut usize, buf: &[u8]) -> Option<()> {
    loop {
        let pos = memchr::memchr2(b':', b'"', &buf[*p..])?;
        *p += pos;
        if buf[*p] == b':' {
            *p += 1;
            while *p < buf.len() && matches!(buf[*p], b' ' | b'\t' | b'\n' | b'\r') {
                *p += 1;
            }
            return Some(());
        }
        // It's a quote — skip the string
        *p += 1;
        let end = memchr::memchr(b'"', &buf[*p..])?;
        *p += end + 1;
    }
}

/// Skip a quoted string value
#[inline]
fn skip_string(p: &mut usize, buf: &[u8]) -> Option<()> {
    if *p < buf.len() && buf[*p] == b'"' {
        *p += 1;
    }
    let end = memchr::memchr(b'"', &buf[*p..])?;
    *p += end + 1;
    Some(())
}

// ---------- Scalar scanners (manual, no str::parse) ----------

/// Manual f32 scanner — handles negatives, digits, fractional part, exponent
#[inline]
fn scan_f32(p: &mut usize, buf: &[u8]) -> f32 {
    let (v, len) = parse_f32(&buf[*p..]);
    *p += len;
    v
}

/// Manual u32 scanner — fastest path for integers
#[inline]
fn scan_u32(p: &mut usize, buf: &[u8]) -> u32 {
    let mut v = 0u32;
    while *p < buf.len() && buf[*p].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((buf[*p] - b'0') as u32);
        *p += 1;
    }
    v
}

#[inline]
fn scan_bool(p: &mut usize, buf: &[u8]) -> bool {
    let is_true = *p < buf.len() && buf[*p] == b't';
    *p += if is_true { 4 } else { 5 };
    is_true
}

#[inline]
fn scan_string<'a>(p: &mut usize, buf: &'a [u8]) -> Option<&'a [u8]> {
    if *p >= buf.len() || buf[*p] != b'"' {
        return None;
    }
    *p += 1;
    let start = *p;
    let end = memchr::memchr(b'"', &buf[start..])?;
    *p = start + end + 1;
    Some(&buf[start..start + end])
}

/// MCC is a quoted digit string, e.g. "5411"
#[inline]
fn scan_mcc(p: &mut usize, buf: &[u8]) -> u32 {
    if *p < buf.len() && buf[*p] == b'"' {
        *p += 1;
    }
    let mut v = 0u32;
    while *p < buf.len() && buf[*p].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((buf[*p] - b'0') as u32);
        *p += 1;
    }
    if *p < buf.len() && buf[*p] == b'"' {
        *p += 1;
    }
    v
}

/// Parse ISO 8601 timestamp: "2024-01-15T10:30:00Z"
#[inline]
fn scan_iso(p: &mut usize, buf: &[u8]) -> Option<(u16, u8, u8, u8, u8)> {
    if *p < buf.len() && buf[*p] == b'"' {
        *p += 1;
    }
    let s = &buf[*p..];
    if s.len() < 20 {
        return None;
    }
    let y = (s[0] - b'0') as u16 * 1000
        + (s[1] - b'0') as u16 * 100
        + (s[2] - b'0') as u16 * 10
        + (s[3] - b'0') as u16;
    let mo = (s[5] - b'0') * 10 + (s[6] - b'0');
    let d = (s[8] - b'0') * 10 + (s[9] - b'0');
    let h = (s[11] - b'0') * 10 + (s[12] - b'0');
    let mi = (s[14] - b'0') * 10 + (s[15] - b'0');
    *p += 20;
    if let Some(off) = memchr::memchr(b'"', &buf[*p..]) {
        *p += off + 1;
    }
    Some((y, mo, d, h, mi))
}

// ---------- parse_f32 — manual, handles digits/fraction/exponent ----------

fn parse_f32(s: &[u8]) -> (f32, usize) {
    let mut pos = 0;
    let mut neg = false;
    if pos < s.len() && s[pos] == b'-' {
        neg = true;
        pos += 1;
    }
    let mut int_part: u64 = 0;
    while pos < s.len() && s[pos].is_ascii_digit() {
        int_part = int_part.wrapping_mul(10).wrapping_add((s[pos] - b'0') as u64);
        pos += 1;
    }
    let mut v = int_part as f64;
    if pos < s.len() && s[pos] == b'.' {
        pos += 1;
        let frac_start = pos;
        let mut frac: u64 = 0;
        while pos < s.len() && s[pos].is_ascii_digit() {
            if pos - frac_start < 18 {
                frac = frac * 10 + (s[pos] - b'0') as u64;
            }
            pos += 1;
        }
        let digits = (pos - frac_start).min(18);
        v += frac as f64 * FRAC_POWERS[digits];
    }
    // Handle optional exponent (e.g., 1.5e3)
    if pos < s.len() && (s[pos] == b'e' || s[pos] == b'E') {
        pos += 1;
        let mut esign = 1i32;
        if pos < s.len() && (s[pos] == b'+' || s[pos] == b'-') {
            if s[pos] == b'-' {
                esign = -1;
            }
            pos += 1;
        }
        let mut e = 0i32;
        while pos < s.len() && s[pos].is_ascii_digit() {
            e = e * 10 + (s[pos] - b'0') as i32;
            pos += 1;
        }
        v *= 10f64.powi(esign * e);
    }
    if neg {
        v = -v;
    }
    (v as f32, pos)
}

// ---------- Date utilities ----------

fn day_of_week(y: u16, m: u8, d: u8) -> u8 {
    const T: [u16; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let ya = if m < 3 { (y - 1) as u32 } else { y as u32 };
    let dow = (ya + ya / 4 - ya / 100 + ya / 400 + T[(m - 1) as usize] as u32 + d as u32) % 7;
    ((dow + 6) % 7) as u8
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
    y1: u16, mo1: u8, d1: u8, h1: u8, mi1: u8,
    y2: u16, mo2: u8, d2: u8, h2: u8, mi2: u8,
) -> u32 {
    let d1 = days_since_epoch(y1 as i32, mo1 as u32, d1 as u32);
    let d2 = days_since_epoch(y2 as i32, mo2 as u32, d2 as u32);
    let m1 = d1 * 1440 + h1 as i64 * 60 + mi1 as i64;
    let m2 = d2 * 1440 + h2 as i64 * 60 + mi2 as i64;
    (m2 - m1).max(0) as u32
}
