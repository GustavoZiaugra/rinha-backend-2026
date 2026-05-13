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

    // Track which top-level fields we've encountered
    let mut in_transaction = false;
    let mut in_customer = false;
    let mut in_merchant = false;
    let mut in_terminal = false;
    let mut in_last_tx = false;
    let mut in_known_merchants = false;
    let mut encountered_known_merchants = false;
    let mut known_merchant_found = false;

    loop {
        skip_ws(buf, &mut pos);
        if pos >= buf.len() { return None; }
        if buf[pos] == b'}' {
            // End of current object
            pos += 1;
            if in_transaction { in_transaction = false; continue; }
            if in_customer { in_customer = false; continue; }
            if in_merchant { in_merchant = false; continue; }
            if in_terminal { in_terminal = false; continue; }
            if in_last_tx { in_last_tx = false; has_last_tx = true; continue; }
            if in_known_merchants { in_known_merchants = false; continue; }
            break; // end of root object
        }

        if buf[pos] == b',' { pos += 1; skip_ws(buf, &mut pos); }
        if pos >= buf.len() { return None; }

        // Parse string key
        if buf[pos] != b'"' { return None; }
        pos += 1;
        let key_start = pos;
        while pos < buf.len() && buf[pos] != b'"' {
            if buf[pos] == b'\\' { pos += 1; } // skip escaped char
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
                b"amount" => { amount = parse_float(buf, &mut pos)?; }
                b"installments" => { installments = parse_int(buf, &mut pos)? as u8; }
                b"requested_at" => {
                    let s = parse_string_raw(buf, &mut pos)?;
                    (req_y, req_mo, req_d, req_h, req_min) = parse_iso(s)?;
                }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else if in_customer {
            match key {
                b"avg_amount" => { customer_avg_amount = parse_float(buf, &mut pos)?; }
                b"tx_count_24h" => { tx_count_24h = parse_int(buf, &mut pos)? as u32; }
                b"known_merchants" => {
                    encountered_known_merchants = true;
                    skip_ws(buf, &mut pos);
                    if pos >= buf.len() || buf[pos] != b'[' { return None; }
                    pos += 1;
                    in_known_merchants = true;
                    // Parse array until ]
                    loop {
                        skip_ws(buf, &mut pos);
                        if pos >= buf.len() { return None; }
                        if buf[pos] == b']' { pos += 1; break; }
                        if buf[pos] == b',' { pos += 1; continue; }
                        let s = parse_string_raw(buf, &mut pos)?;
                        if s == &merchant_id_buf[..merchant_id_len] {
                            known_merchant_found = true;
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
                b"avg_amount" => { merchant_avg_amount = parse_float(buf, &mut pos)?; }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else if in_terminal {
            match key {
                b"is_online" => { is_online = parse_bool(buf, &mut pos)?; }
                b"card_present" => { card_present = parse_bool(buf, &mut pos)?; }
                b"km_from_home" => { km_from_home = parse_float(buf, &mut pos)?; }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else if in_last_tx {
            match key {
                b"timestamp" => {
                    let s = parse_string_raw(buf, &mut pos)?;
                    (lt_y, lt_mo, lt_d, lt_h, lt_min) = parse_iso(s)?;
                }
                b"km_from_current" => { km_from_current = parse_float(buf, &mut pos)?; }
                _ => { skip_value(buf, &mut pos)?; }
            }
        } else {
            // Top-level fields
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
                        // null
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

    // Compute minutes and day_of_week
    hour = req_h as u8;
    day_of_week = crate::json_utils::day_of_week(req_y, req_mo, req_d);

    if has_last_tx {
        minutes_since_last = crate::json_utils::minutes_between(
            lt_y, lt_mo, lt_d, lt_h, lt_min,
            req_y, req_mo, req_d, req_h, req_min,
        );
    }

    // known_merchant: if we encountered the array and merchant wasn't found, it's unknown
    is_unknown_merchant = encountered_known_merchants && !known_merchant_found;

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
        hour,
        day_of_week,
        is_online,
        card_present,
        is_unknown_merchant,
        has_last_tx,
    })
}

// ---------- low-level parsers ----------

fn skip_ws(buf: &[u8], pos: &mut usize) {
    while *pos < buf.len() && buf[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

fn parse_float(buf: &[u8], pos: &mut usize) -> Option<f32> {
    let mut neg = false;
    skip_ws(buf, pos);
    if *pos >= buf.len() { return None; }
    if buf[*pos] == b'-' { neg = true; *pos += 1; }
    let start = *pos;
    while *pos < buf.len() && (buf[*pos].is_ascii_digit() || buf[*pos] == b'.') {
        *pos += 1;
    }
    if *pos == start { return None; }
    let s = std::str::from_utf8(&buf[start..*pos]).ok()?;
    let v: f32 = s.parse().ok()?;
    Some(if neg { -v } else { v })
}

fn parse_int(buf: &[u8], pos: &mut usize) -> Option<u32> {
    skip_ws(buf, pos);
    let start = *pos;
    while *pos < buf.len() && buf[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start { return None; }
    let s = std::str::from_utf8(&buf[start..*pos]).ok()?;
    s.parse().ok()
}

fn parse_bool(buf: &[u8], pos: &mut usize) -> Option<bool> {
    skip_ws(buf, pos);
    if buf.get(*pos..*pos+4) == Some(b"true") {
        *pos += 4;
        Some(true)
    } else if buf.get(*pos..*pos+5) == Some(b"false") {
        *pos += 5;
        Some(false)
    } else {
        None
    }
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
    *pos += 1; // closing "
    Some(s)
}

fn parse_mcc(s: &[u8]) -> Option<u32> {
    std::str::from_utf8(s).ok()?.parse().ok()
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

/// Skip over a JSON value entirely (doesn't matter what type)
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
            // number, true, false, null — consume alphanumeric
            while *pos < buf.len() && (buf[*pos].is_ascii_alphanumeric() || buf[*pos] == b'.' || buf[*pos] == b'-') {
                *pos += 1;
            }
        }
    }
    Some(())
}
