use crate::vector::Payload as VecPayload;

/// Manual byte-level JSON parser — no heap allocations.
/// The payload is ~280-400 bytes of known structure. We parse
/// field-by-field, matching keys as byte strings, extracting
/// values directly into stack locals, then construct the final Payload.
///
/// This avoids serde_json's String/Vec allocations and is ~10x faster.

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

    // Collect known_merchant IDs separately because merchant.id appears AFTER customer.known_merchants
    // in the JSON payload. We store them here and match against merchant_id after all parsing.
    let mut known_ids: [[u8; 64]; 10] = [[0; 64]; 10];
    let mut known_lens: [usize; 10] = [0; 10];
    let mut known_count = 0usize;

    // Track which top-level fields we've encountered
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
                _ => { skip_value(buf, &mut pos); }
            }
        } else if in_customer {
            match key {
                b"avg_amount" => { customer_avg_amount = scan_f32(buf, &mut pos); }
                b"km_from_home" => { km_from_home = scan_f32(buf, &mut pos); }
                b"km_from_current" => { km_from_current = scan_f32(buf, &mut pos); }
                b"tx_count_24h" => { tx_count_24h = scan_u32(buf, &mut pos); }
                b"minutes_since_last" => { minutes_since_last = scan_u32(buf, &mut pos); }
                b"known_merchants" => {
                    in_known_merchants = true;
                    encountered_known_merchants = true;
                    // skip opening [
                    skip_ws(buf, &mut pos);
                    if pos < buf.len() && buf[pos] == b'[' { pos += 1; }
                }
                _ => { skip_value(buf, &mut pos); }
            }
        } else if in_known_merchants {
            if buf[pos] == b']' {
                pos += 1;
                in_known_merchants = false;
            } else if buf[pos] == b',' {
                pos += 1;
                skip_ws(buf, &mut pos);
            } else if buf[pos] == b'"' {
                pos += 1;
                let id_start = pos;
                while pos < buf.len() && buf[pos] != b'"' { pos += 1; }
                if pos < buf.len() {
                    let id_len = pos - id_start;
                    if known_count < 10 && id_len <= 64 {
                        known_ids[known_count][..id_len].copy_from_slice(&buf[id_start..pos]);
                        known_lens[known_count] = id_len;
                        known_count += 1;
                    }
                    pos += 1; // skip closing "
                }
                skip_ws(buf, &mut pos);
            } else {
                skip_value(buf, &mut pos);
            }
        } else if in_merchant {
            match key {
                b"id" => {
                    if buf[pos] == b'"' {
                        pos += 1;
                        merchant_id_len = 0;
                        while pos < buf.len() && buf[pos] != b'"' && merchant_id_len < 64 {
                            merchant_id_buf[merchant_id_len] = buf[pos];
                            merchant_id_len += 1;
                            pos += 1;
                        }
                        if pos < buf.len() { pos += 1; }
                    }
                }
                b"avg_amount" => { merchant_avg_amount = scan_f32(buf, &mut pos); }
                _ => { skip_value(buf, &mut pos); }
            }
        } else if in_terminal {
            match key {
                b"mcc" => { mcc = scan_u32(buf, &mut pos); }
                _ => { skip_value(buf, &mut pos); }
            }
        } else if in_last_tx {
            match key {
                b"amount" => {} // skip — used for last_tx amount? not needed for vector
                b"km_from_current" => { km_from_current = scan_f32(buf, &mut pos); }
                b"requested_at" => {
                    let s = parse_string_raw(buf, &mut pos)?;
                    (lt_y, lt_mo, lt_d, lt_h, lt_min) = parse_iso(s)?;
                }
                _ => { skip_value(buf, &mut pos); }
            }
        } else {
            // Top-level fields
            match key {
                b"transaction" => { in_transaction = true; }
                b"customer" => { in_customer = true; }
                b"merchant" => { in_merchant = true; }
                b"terminal" => { in_terminal = true; }
                b"last_tx" => { in_last_tx = true; }
                b"is_online" => { is_online = scan_bool(buf, &mut pos); }
                b"card_present" => { card_present = scan_bool(buf, &mut pos); }
                b"is_unknown_merchant" => { is_unknown_merchant = scan_bool(buf, &mut pos); }
                _ => { skip_value(buf, &mut pos); }
            }
        }
    }

    // Check if merchant_id matches any known merchant
    if !encountered_known_merchants || merchant_id_len == 0 {
        is_unknown_merchant = false;
    } else {
        let mut found = false;
        for i in 0..known_count {
            if known_lens[i] == merchant_id_len
                && &known_ids[i][..merchant_id_len] == &merchant_id_buf[..merchant_id_len]
            {
                found = true;
                break;
            }
        }
        is_unknown_merchant = !found;
    }

    // Compute minutes_since_last from timestamps if has_last_tx
    if !has_last_tx {
        minutes_since_last = 0;
    } else {
        // Both timestamps are ISO format in UTC (no timezone)
        let total_req = req_y as u64 * 525600 + req_mo as u64 * 43200 + req_d as u64 * 1440
            + req_h as u64 * 60 + req_min as u64;
        let total_lt = lt_y as u64 * 525600 + lt_mo as u64 * 43200 + lt_d as u64 * 1440
            + lt_h as u64 * 60 + lt_min as u64;
        if total_req > total_lt {
            let diff = (total_req - total_lt) as u32;
            minutes_since_last = diff.min(u32::MAX);
        }
    }

    Some(VecPayload {
        amount,
        customer_avg_amount,
        merchant_avg_amount,
        km_from_home,
        km_from_current,
        tx_count_24h,
        mcc,
        minutes_since_last,
        installments,
        hour: req_h as u8,
        day_of_week: req_d as u8 % 7,
        is_online,
        card_present,
        is_unknown_merchant,
        has_last_tx,
    })
}

#[inline(always)]
fn skip_ws(buf: &[u8], pos: &mut usize) {
    while *pos < buf.len() && buf[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

#[inline(always)]
fn scan_u32(buf: &[u8], pos: &mut usize) -> u32 {
    let mut val: u32 = 0;
    while *pos < buf.len() && buf[*pos].is_ascii_digit() {
        val = val.wrapping_mul(10).wrapping_add((buf[*pos] - b'0') as u32);
        *pos += 1;
    }
    val
}

#[inline(always)]
fn scan_f32(buf: &[u8], pos: &mut usize) -> f32 {
    // Parse optional sign
    let neg = if *pos < buf.len() && buf[*pos] == b'-' {
        *pos += 1;
        true
    } else {
        false
    };

    // Integer part
    let int_val: u32 = {
        let mut v = 0u32;
        while *pos < buf.len() && buf[*pos].is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((buf[*pos] - b'0') as u32);
            *pos += 1;
        }
        v
    };

    // Fractional part
    let mut frac_val: f32 = 0.0;
    if *pos < buf.len() && buf[*pos] == b'.' {
        *pos += 1;
        let mut divisor = 1.0f32;
        while *pos < buf.len() && buf[*pos].is_ascii_digit() {
            frac_val = frac_val * 10.0 + (buf[*pos] - b'0') as f32;
            divisor *= 10.0;
            *pos += 1;
        }
        frac_val /= divisor;
    }

    let mut result = int_val as f32 + frac_val;
    if neg { result = -result; }
    result
}

#[inline(always)]
fn scan_bool(buf: &[u8], pos: &mut usize) -> bool {
    if *pos + 3 < buf.len() && &buf[*pos..*pos+4] == b"true" {
        *pos += 4;
        return true;
    }
    if *pos + 4 < buf.len() && &buf[*pos..*pos+5] == b"false" {
        *pos += 5;
    }
    false
}

#[inline(always)]
fn parse_string_raw<'a>(buf: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    if *pos >= buf.len() || buf[*pos] != b'"' { return None; }
    *pos += 1;
    let start = *pos;
    while *pos < buf.len() && buf[*pos] != b'"' {
        if buf[*pos] == b'\\' { *pos += 1; }
        *pos += 1;
    }
    if *pos >= buf.len() { return None; }
    let s = &buf[start..*pos];
    *pos += 1; // skip closing "
    Some(s)
}

/// Parse ISO-like datetime: "2025-03-15T14:30:00.000Z"
/// Returns (year, month, day, hour, minute)
fn parse_iso(s: &[u8]) -> Option<(u32, u32, u32, u32, u32)> {
    if s.len() < 16 { return None; }
    let y = parse_digits_4(&s[0..4])?;
    let mo = parse_digits_2(&s[5..7])?;
    let d = parse_digits_2(&s[8..10])?;
    let h = parse_digits_2(&s[11..13])?;
    let min = parse_digits_2(&s[14..16])?;
    Some((y, mo, d, h, min))
}

#[inline(always)]
fn parse_digits_4(s: &[u8]) -> Option<u32> {
    if s.len() < 4 { return None; }
    let d0 = (s[0] - b'0') as u32;
    let d1 = (s[1] - b'0') as u32;
    let d2 = (s[2] - b'0') as u32;
    let d3 = (s[3] - b'0') as u32;
    if d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9 { return None; }
    Some(d0 * 1000 + d1 * 100 + d2 * 10 + d3)
}

#[inline(always)]
fn parse_digits_2(s: &[u8]) -> Option<u32> {
    if s.len() < 2 { return None; }
    let d0 = (s[0] - b'0') as u32;
    let d1 = (s[1] - b'0') as u32;
    if d0 > 9 || d1 > 9 { return None; }
    Some(d0 * 10 + d1)
}

/// Skip over a JSON value (number, string, object, array, bool, null)
fn skip_value(buf: &[u8], pos: &mut usize) {
    skip_ws(buf, pos);
    if *pos >= buf.len() { return; }
    match buf[*pos] {
        b'"' => {
            *pos += 1;
            while *pos < buf.len() && buf[*pos] != b'"' {
                if buf[*pos] == b'\\' { *pos += 1; }
                *pos += 1;
            }
            if *pos < buf.len() { *pos += 1; }
        }
        b'{' | b'[' => {
            let open = buf[*pos];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 1u32;
            *pos += 1;
            while *pos < buf.len() && depth > 0 {
                if buf[*pos] == b'"' {
                    *pos += 1;
                    while *pos < buf.len() && buf[*pos] != b'"' {
                        if buf[*pos] == b'\\' { *pos += 1; }
                        *pos += 1;
                    }
                    if *pos < buf.len() { *pos += 1; }
                } else {
                    if buf[*pos] == close { depth -= 1; }
                    if buf[*pos] == open { depth += 1; }
                    *pos += 1;
                }
            }
        }
        _ => {
            // number, true, false, null — consume alphanumeric
            while *pos < buf.len()
                && (buf[*pos].is_ascii_alphanumeric() || buf[*pos] == b'.' || buf[*pos] == b'-')
            {
                *pos += 1;
            }
        }
    }
}
