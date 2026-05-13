/// Date/time utilities extracted from the old serde-based json module.

/// Sakamoto day-of-week: returns Mon=0..Sun=6 (bot-compatible offset applied)
pub fn day_of_week(y: u32, m: u32, d: u32) -> u8 {
    let t = [0u32, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let ya = if m < 3 { y - 1 } else { y };
    let dow = (ya + ya / 4 - ya / 100 + ya / 400 + t[(m - 1) as usize] + d) % 7;
    ((dow + 6) % 7) as u8
}

/// minutes_between(last_tx_datetime, request_datetime)
pub fn minutes_between(
    y1: u32, mo1: u32, d1: u32, h1: u32, min1: u32,
    y2: u32, mo2: u32, d2: u32, h2: u32, min2: u32,
) -> u32 {
    let d1 = days_since_epoch(y1 as i32, mo1, d1);
    let d2 = days_since_epoch(y2 as i32, mo2, d2);
    let m1 = d1 * 1440 + h1 as i64 * 60 + min1 as i64;
    let m2 = d2 * 1440 + h2 as i64 * 60 + min2 as i64;
    (m2 - m1).max(0) as u32
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
