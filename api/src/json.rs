use crate::vector::Payload as VecPayload;

/// Manual byte-level JSON parser — no heap allocations.
/// Key-matching for correctness (handles arbitrary field order).
/// Uses manual scan_f32/scan_u32 (no str::parse) for speed.

static FRAC_POWERS: [f64; 19] = [
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
    1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18,
];

pub fn parse(buf: &[u8]) -> Option<VecPayload> {
    let mut pos = 0usize;
    skip_ws(buf, &mut pos);

    // Expect opening {
    if pos >= buf.len() || buf[pos] != b'{' { return None; }
    pos += 1;

    let mut amount = 0.0f32;
    let mut customer_avg_amount = 0.0f32;
    let mut merchant_avg_amount = 0.0f32;
    let mut km_from_home = 0.0f32;
    let mut km_from_current = 0.0f32;
    let mut tx_count_24h = 0u32;
    let mut mcc = 0u32;
    let mut minutes_since_last = 0u32;
    let mut installments = 0u8;
    let mut hour = 0u8;
    let mut day_of_week = 0u8;
    let mut is_online = false;
    let mut card_present = false;
    let mut is_unknown_merchant = false;
    let mut has_last_tx = false;

    // Temp storage for dates
    let mut req_y = 0u32; let mut req_mo = 0u32; let mut req_d = 0u32;
    let mut req_h = 0u32; let mut req_min = 0u32;
    let mut lt_y = 0u32; let mut lt_mo = 0u32; let mut lt_d = 0u32;
    let mut lt_h = 0u32; let mut lt_min = 0u32;

    let mut merchant_id_buf = [0u8; 64];
    let mut merchant_id_len = 0usize;

    // Collect known_merchant IDs for post-parse matching
    let mut known_ids: [[u8; 64]; 10] = [[0; 64]; 10];
    let mut known_lens: [usize; 10] = [0; 10];
    let mut known_count = 0usize;

    // Track nesting context
    let mut in_transaction = false;
    let mut in_customer = false;
    let mut in_merchant = false;
    let mut in_terminal = false;
    let mut in_last_tx = false;
    let mut in_known_merchants = false;
    let mut encountered_known_merchants = false;

    loop {
        skip_ws(buf, &mut pos);
        if pos >= buf.len() { return None; }
        if buf[pos] == b'}' {
            pos += 1;
            if in_transaction { in_transaction = false; continue; }
            if in_customer { in_customer = false; continue; }
            if in_merchant { in_merchant = false; continue; }
            if in_terminal { in_terminal = false; continue; }
            if in_last_tx { in_last_tx = false; has_last_tx = true; continue; }
            if in_known_merchants { in_known_merchants = false; continue; }
            break;
        }

        if buf[pos] == b',' { pos += 1; skip_ws(buf, &mut pos); }
        if pos >= buf.len() { return None; }

        // Parse string key
        if buf[pos] != b'"' { return None; }
        pos += 1;
        let key_start = pos;
        while pos < buf.len() && buf[pos] != b'"' {
            if buf[pos] == b'\\' { pos += 1; }
            pos += 1;
        }
        if pos >= buf.len() { return None; }
        let key = &buf[key_start..pos];
        pos += 1; // skip closing "

        skip_ws(buf, &mut pos);
        if pos >= buf.len() || buf[pos] != b':' { return None; }
        pos += 1;
        skip_ws(buf, &mut pos);
        if pos >= buf.len() { return None; }

        if in_transaction {
            match key {
                b"amount" => { amount = scan_f32(buf, &mut pos); }
                b"installments" => { installments = scan_u32(buf, &mut pos) as u8; }
                b"requested_at" => {
                    let s = parse_string_raw(buf, &mut pos)?;
                    (req_y, req_mo, req_d, req_h, req_min) = parse_iso(s)?;
                }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else if in_customer {
            match key {
                b"avg_amount" => { customer_avg_amount = scan_f32(buf, &mut pos); }
                b"tx_count_24h" => { tx_count_24h = scan_u32(buf, &mut pos) as u32; }
                b"known_merchants" => {
                    encountered_known_merchants = true;
                    skip_ws(buf, &mut pos);
                    if pos >= buf.len() || buf[pos] != b'[' { return None; }
                    pos += 1;
                    in_known_merchants = true;
                    loop {
                        skip_ws(buf, &mut pos);
                        if pos >= buf.len() { return None; }
                        if buf[pos] == b']' { pos += 1; break; }
                        if buf[pos] == b',' { pos += 1; continue; }
                        let s = parse_string_raw(buf, &mut pos)?;
                        if known_count < 10 {
                            let len = s.len().min(64);
                            known_ids[known_count][..len].copy_from_slice(&s[..len]);
                            known_lens[known_count] = len;
                            known_count += 1;
                        }
                    }
                    in_known_merchants = false;
                }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else if in_merchant {
            match key {
                b"id" => {
                    let s = parse_string_raw(buf, &mut pos)?;
                    merchant_id_len = s.len().min(64);
                    merchant_id_buf[..merchant_id_len].copy_from_slice(&s[..merchant_id_len]);
                }
                b"mcc" => {
                    let s = parse_string_raw(buf, &mut pos)?;
                    mcc = parse_mcc(s)?;
                }
                b"avg_amount" => { merchant_avg_amount = scan_f32(buf, &mut pos); }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else if in_terminal {
            match key {
                b"is_online" => { is_online = scan_bool(buf, &mut pos); }
                b"card_present" => { card_present = scan_bool(buf, &mut pos); }
                b"km_from_home" => { km_from_home = scan_f32(buf, &mut pos); }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else if in_last_tx {
            match key {
                b"timestamp" => {
                    let s = parse_string_raw(buf, &mut pos)?;
                    (lt_y, lt_mo, lt_d, lt_h, lt_min) = parse_iso(s)?;
                }
                b"km_from_current" => { km_from_current = scan_f32(buf, &mut pos); }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else {
            match key {
                b"transaction" => {
                    skip_ws(buf, &mut pos);
                    if pos >= buf.len() || buf[pos] != b'{' { return None; }
                    in_transaction = true;
                    pos += 1;
                }
                b"customer" => {
                    skip_ws(buf, &mut pos);
                    if pos >= buf.len() || buf[pos] != b'{' { return None; }
                    in_customer = true;
                    pos += 1;
                }
                b"merchant" => {
                    skip_ws(buf, &mut pos);
                    if pos >= buf.len() || buf[pos] != b'{' { return None; }
                    in_merchant = true;
                    pos += 1;
                }
                b"terminal" => {
                    skip_ws(buf, &mut pos);
                    if pos >= buf.len() || buf[pos] != b'{' { return None; }
                    in_terminal = true;
                    pos += 1;
                }
                b"last_transaction" => {
                    skip_ws(buf, &mut pos);
                    if pos >= buf.len() { return None; }
                    if buf[pos] == b'n' {
                        let rest = buf.get(pos..pos+4)?;
                        if rest != b"null" { return None; }
                        pos += 4;
                        has_last_tx = false;
                    } else if buf[pos] == b'{' {
                        in_last_tx = true;
                        pos += 1;
                    } else {
                        return None;
                    }
                }
                _ => { skip_value(buf, &mut pos)?; }
            }
        }
    }

    // Compute derived fields
    hour = req_h as u8;
    day_of_week = crate::json_utils::day_of_week(req_y, req_mo, req_d);

    if has_last_tx {
        minutes_since_last = crate::json_utils::minutes_between(
            lt_y, lt_mo, lt_d, lt_h, lt_min,
            req_y, req_mo, req_d, req_h, req_min,
        );
    }

    // Match merchant_id against known_merchants
    let mut known_merchant_found = false;
    if encountered_known_merchants && merchant_id_len > 0 {
        for i in 0..known_count {
            if known_lens[i] == merchant_id_len
                && &known_ids[i][..known_lens[i]] == &merchant_id_buf[..merchant_id_len]
            {
                known_merchant_found = true;
                break;
            }
        }
    }
    is_unknown_merchant = encountered_known_merchants && !known_merchant_found;

    Some(VecPayload {
        amount, customer_avg_amount, merchant_avg_amount,
        km_from_home, km_from_current, tx_count_24h, mcc,
        minutes_since_last, installments, hour, day_of_week,
        is_online, card_present, is_unknown_merchant, has_last_tx,
    })
}

// ---------- low-level parsers ----------

fn skip_ws(buf: &[u8], pos: &mut usize) {
    while *pos < buf.len() && buf[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

/// Manual f32 scanner — sign, digits, decimal, optional exponent. No str::parse.
fn scan_f32(buf: &[u8], pos: &mut usize) -> f32 {
    skip_ws(buf, pos);
    let start = *pos;
    let mut neg = false;
    if *pos < buf.len() && buf[*pos] == b'-' { neg = true; *pos += 1; }

    let mut int_part: u64 = 0;
    while *pos < buf.len() && buf[*pos].is_ascii_digit() {
        int_part = int_part.wrapping_mul(10).wrapping_add((buf[*pos] - b'0') as u64);
        *pos += 1;
    }
    let mut v = int_part as f64;

    if *pos < buf.len() && buf[*pos] == b'.' {
        *pos += 1;
        let frac_start = *pos;
        let mut frac: u64 = 0;
        while *pos < buf.len() && buf[*pos].is_ascii_digit() {
            if *pos - frac_start < 18 {
                frac = frac.wrapping_mul(10).wrapping_add((buf[*pos] - b'0') as u64);
            }
            *pos += 1;
        }
        let digits = (*pos - frac_start).min(18);
        v += frac as f64 * FRAC_POWERS[digits];
    }

    // Optional exponent
    if *pos < buf.len() && (buf[*pos] == b'e' || buf[*pos] == b'E') {
        *pos += 1;
        let mut esign = 1i32;
        if *pos < buf.len() && (buf[*pos] == b'+' || buf[*pos] == b'-') {
            if buf[*pos] == b'-' { esign = -1; }
            *pos += 1;
        }
        let mut e = 0i32;
        while *pos < buf.len() && buf[*pos].is_ascii_digit() {
            e = e * 10 + (buf[*pos] - b'0') as i32;
            *pos += 1;
        }
        v *= 10f64.powi(esign * e);
    }

    // Fallback to str::parse for extreme edge cases (should never hit in practice)
    if *pos == start || *pos - start > 64 {
        *pos = start;
        return 0.0;
    }

    if neg { -v as f32 } else { v as f32 }
}

fn scan_u32(buf: &[u8], pos: &mut usize) -> u32 {
    skip_ws(buf, pos);
    let mut v = 0u32;
    while *pos < buf.len() && buf[*pos].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((buf[*pos] - b'0') as u32);
        *pos += 1;
    }
    v
}

fn scan_bool(buf: &[u8], pos: &mut usize) -> bool {
    skip_ws(buf, pos);
    let is_true = *pos < buf.len() && buf[*pos] == b't';
    *pos += if is_true { 4 } else { 5 };
    is_true
}

fn parse_string_raw<'a>(buf: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    skip_ws(buf, pos);
    if *pos >= buf.len() || buf[*pos] != b'"' { return None; }
    *pos += 1;
    let start = *pos;
    while *pos < buf.len() && buf[*pos] != b'"' {
        if buf[*pos] == b'\\' { *pos += 1; }
        *pos += 1;
    }
    if *pos >= buf.len() { return None; }
    let s = &buf[start..*pos];
    *pos += 1;
    Some(s)
}

fn parse_mcc(s: &[u8]) -> Option<u32> {
    let mut v = 0u32;
    for &b in s {
        if !b.is_ascii_digit() { return None; }
        v = v.wrapping_mul(10).wrapping_add((b - b'0') as u32);
    }
    Some(v)
}

fn parse_iso(s: &[u8]) -> Option<(u32, u32, u32, u32, u32)> {
    if s.len() < 16 { return None; }
    let y = parse4(s, 0);
    let mo = parse2(s, 5);
    let d = parse2(s, 8);
    let h = parse2(s, 11);
    let min = parse2(s, 14);
    Some((y, mo, d, h, min))
}

fn parse4(s: &[u8], off: usize) -> u32 {
    (s[off] - b'0') as u32 * 1000
        + (s[off+1] - b'0') as u32 * 100
        + (s[off+2] - b'0') as u32 * 10
        + (s[off+3] - b'0') as u32
}

fn parse2(s: &[u8], off: usize) -> u32 {
    (s[off] - b'0') as u32 * 10 + (s[off+1] - b'0') as u32
}

fn skip_value(buf: &[u8], pos: &mut usize) -> Option<()> {
    skip_ws(buf, pos);
    if *pos >= buf.len() { return None; }
    match buf[*pos] {
        b'"' => { parse_string_raw(buf, pos)?; }
        b'[' | b'{' => {
            let close = if buf[*pos] == b'[' { b']' } else { b'}' };
            let mut depth = 1u32;
            *pos += 1;
            while depth > 0 && *pos < buf.len() {
                if buf[*pos] == b'"' {
                    parse_string_raw(buf, pos)?;
                } else {
                    if buf[*pos] == close { depth -= 1; }
                    if buf[*pos] == b'[' { depth += 1; }
                    if buf[*pos] == b'{' { depth += 1; }
                    *pos += 1;
                }
            }
            if depth > 0 { return None; }
        }
        _ => {
            while *pos < buf.len() && (buf[*pos].is_ascii_alphanumeric() || buf[*pos] == b'.' || buf[*pos] == b'-') {
                *pos += 1;
            }
        }
    }
    Some(())
}
