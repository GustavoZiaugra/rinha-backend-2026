use crate::data;
use crate::json;
use crate::knn;
use crate::vector;

use memchr::memmem;
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicBool, Ordering};

const RX_CAP: usize = 8192;

/// Thread pool worker count
const WORKERS: usize = 4;

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

/// Handle a single HTTP request, then close the connection.
/// No keep-alive — each connection handles exactly one request.
/// This prevents worker threads from being tied up waiting for
/// the next request on an idle keepalive connection.
fn handle_conn(stream: &mut (impl Read + Write)) {
    let mut buf = [0u8; RX_CAP];
    let mut used = 0usize;

    loop {
        match stream.read(&mut buf[used..]) {
            Ok(0) => break,
            Ok(n) => used += n,
            Err(_) => break,
        }

        if used < 16 {
            continue;
        }

        let header_end = match memmem_find(&buf[..used], b"\r\n\r\n") {
            Some(p) => p,
            None => {
                if used >= RX_CAP {
                    break;
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
        let resp = handle_request(path, body);
        let _ = stream.write_all(resp);

        // Always close after one request — no keepalive
        break;
    }
}

fn handle_request(path: &[u8], body: &[u8]) -> &'static [u8] {
    // /ready ALWAYS returns 200 — health check must pass immediately
    // even before dataset is fully loaded.
    if path == b"/ready" {
        return RESP_READY;
    }
    if !crate::is_ready() {
        return RESP_NOT_READY;
    }
    if path != b"/fraud-score" {
        return RESP_NOT_FOUND;
    }
    let payload = match json::parse(body) {
        Some(p) => p,
        None => return RESP_BAD,
    };
    let query = vector::vectorize(&payload);
    let ds = data::dataset();
    let fraud_count = knn::knn5_fraud_count(&query, ds);
    HTTP_FRAUD[fraud_count.min(5) as usize]
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

/// All responses include Connection: close so each connection
/// handles exactly one request. This prevents worker threads from
/// blocking on idle keepalive connections.
pub const HTTP_FRAUD: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

pub const RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
pub const RESP_NOT_FOUND: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
pub const RESP_BAD: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
pub const RESP_NOT_READY: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
