use crate::data;
use crate::json;
use crate::knn;
use crate::vector;

use memchr::memmem;
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicBool, Ordering};

const RX_CAP: usize = 8192;

/// Thread pool worker count — 8 workers reduce queue wait to near zero on 100 concurrent.
const WORKERS: usize = 8;

/// Flag to signal workers to shut down
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn serve(listener: UnixListener) -> std::io::Result<()> {
    let listener = std::sync::Arc::new(listener);

    let mut handles = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        let l = listener.clone();
        handles.push(std::thread::spawn(move || {
            worker(&l);
        }));
    }

    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn worker(listener: &UnixListener) {
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((mut stream, _addr)) => {
                let _ = stream.set_nonblocking(false);
                handle_conn(&mut stream);
            }
            Err(_) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

/// Handle one HTTP connection — supports keep-alive and request pipelining.
/// Instead of closing after one request, we loop reading requests from the
/// same connection. The bot can pipeline requests, reducing accept/connect
/// overhead and keeping worker caches warm.
fn handle_conn(stream: &mut (impl Read + Write)) {
    let mut buf = [0u8; RX_CAP];
    let mut used = 0usize;

    loop {
        // Read until we have at least headers + content-length bytes
        loop {
            match stream.read(&mut buf[used..]) {
                Ok(0) => return,  // client closed
                Ok(n) => used += n,
                Err(_) => return,
            }

            if used < 16 {
                continue;
            }

            let header_end = match memmem_find(&buf[..used], b"\r\n\r\n") {
                Some(p) => p,
                None => {
                    if used >= RX_CAP {
                        return;
                    }
                    continue;
                }
            };

            let content_len = parse_content_length(&buf[..header_end]).unwrap_or(0);
            let total_len = header_end + 4 + content_len;
            if used < total_len {
                continue;
            }

            let body = &buf[header_end + 4..total_len];
            let path = parse_path(&buf[..header_end]);
            let conn_close = has_connection_close(&buf[..header_end]);

            let resp = match path {
                b"/ready" => RESP_READY,
                _ if !crate::is_ready() => RESP_NOT_READY,
                b"/fraud-score" => {
                    match json::parse(body) {
                        Some(payload) => {
                            let query = vector::vectorize(&payload);
                            let ds = data::dataset();
                            let fraud_count = knn::knn5_fraud_count(&query, ds);
                            if conn_close {
                                HTTP_FRAUD_CLOSE[fraud_count.min(5) as usize]
                            } else {
                                HTTP_FRAUD_KA[fraud_count.min(5) as usize]
                            }
                        }
                        None => if conn_close { RESP_BAD_CLOSE } else { RESP_BAD_KA },
                    }
                }
                _ => if conn_close { RESP_NOT_FOUND_CLOSE } else { RESP_NOT_FOUND_KA },
            };

            let _ = stream.write_all(resp);

            // Shift remaining data for pipelined requests
            let remaining = used - total_len;
            if remaining > 0 {
                buf.copy_within(total_len..used, 0);
            }
            used = remaining;

            if conn_close {
                return;
            }
            // Continue loop for next request
        }
    }
}

/// Detect if the client sent "Connection: close"
fn has_connection_close(headers: &[u8]) -> bool {
    // Look for "Connection:" field — case-insensitive check
    if let Some(mut p) = memmem_find(headers, b"Connection: ")
        .or_else(|| memmem_find(headers, b"connection: "))
    {
        p += 12; // skip past "Connection: "
        // Skip whitespace
        while p < headers.len() && (headers[p] == b' ' || headers[p] == b'\t') {
            p += 1;
        }
        // Check for "close" (case-insensitive)
        if p + 4 < headers.len() {
            let c = headers[p];
            if (c == b'c' || c == b'C')
                && (headers[p+1] == b'l' || headers[p+1] == b'L')
                && (headers[p+2] == b'o' || headers[p+2] == b'O')
                && (headers[p+3] == b's' || headers[p+3] == b'S')
                && (headers[p+4] == b'e' || headers[p+4] == b'E')
            {
                return true;
            }
        }
    }
    false
}

fn memmem_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    memmem::find(haystack, needle)
}

fn parse_path(headers: &[u8]) -> &[u8] {
    let mut i = 0;
    while i < headers.len() && headers[i] != b' ' {
        i += 1;
    }
    if i >= headers.len() {
        return &[];
    }
    i += 1;
    let start = i;
    while i < headers.len() && headers[i] != b' ' {
        i += 1;
    }
    &headers[start..i]
}

fn parse_content_length(headers: &[u8]) -> Option<usize> {
    const CL_LOWER: &[u8] = b"content-length:";
    const CL_TITLE: &[u8] = b"Content-Length:";
    let start = memmem::find(headers, CL_LOWER)
        .or_else(|| memmem::find(headers, CL_TITLE))?;
    let mut p = start + 15;
    while p < headers.len() && headers[p].is_ascii_whitespace() {
        p += 1;
    }
    let mut v = 0usize;
    while p < headers.len() && headers[p].is_ascii_digit() {
        v = v * 10 + (headers[p] - b'0') as usize;
        p += 1;
    }
    Some(v)
}

// ── Response constants ──────────────────────────────────────────────

/// Fraud responses with Connection: keep-alive (HTTP/1.1 default)
pub const HTTP_FRAUD_KA: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

/// Fraud responses with Connection: close (when client requests it)
pub const HTTP_FRAUD_CLOSE: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

/// Non-fraud responses — keep-alive variants (no Connection header → HTTP/1.1 default)
pub const RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
pub const RESP_NOT_FOUND_KA: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
pub const RESP_NOT_FOUND_CLOSE: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
pub const RESP_BAD_KA: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
pub const RESP_BAD_CLOSE: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
pub const RESP_NOT_READY: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
