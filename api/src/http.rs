use crate::data;
use crate::json;
use crate::knn;
use crate::vector;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};

const RX_CAP: usize = 8192;

/// Thread pool worker count (16 = good throughput under 0.45 CPU, I/O-bound workers share CPU)
const WORKERS: usize = 16;

/// Flag to signal workers to shut down
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn serve(listener: TcpListener) -> std::io::Result<()> {
    // Spawn N worker threads, each accepting from the shared listener.
    // This avoids the overhead of spawning one thread per connection.
    let listener = std::sync::Arc::new(listener);

    let mut handles = Vec::with_capacity(WORKERS);
    for id in 0..WORKERS {
        let l = listener.clone();
        handles.push(std::thread::spawn(move || {
            worker(id, &l);
        }));
    }

    // Wait for all workers to finish (shouldn't happen in normal operation)
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn worker(id: usize, listener: &TcpListener) {
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((mut stream, addr)) => {
                let _ = stream.set_nonblocking(false);
                handle_conn(&mut stream);
            }
            Err(_) => {
                // On accept error, briefly yield so we don't spin
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

fn handle_conn(stream: &mut TcpStream) {
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

        // Check if keep-alive
        let is_keepalive = is_keepalive(&buf[..header_end]);
        if !is_keepalive {
            break;
        }

        // Shift remaining data
        if used > total_len {
            buf.copy_within(total_len..used, 0);
            used -= total_len;
        } else {
            used = 0;
        }
    }
}

fn handle_request(path: &[u8], body: &[u8]) -> &'static [u8] {
    if !crate::is_ready() {
        return RESP_NOT_READY;
    }
    if path == b"/ready" {
        return RESP_READY;
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
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }
    for i in 0..=haystack.len() - needle.len() {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

fn parse_path(headers: &[u8]) -> &[u8] {
    // headers starts with "METHOD PATH HTTP/1.1\r\n..."
    let mut i = 0;
    while i < headers.len() && headers[i] != b' ' {
        i += 1;
    }
    if i >= headers.len() {
        return &[];
    }
    i += 1; // skip space
    let start = i;
    while i < headers.len() && headers[i] != b' ' {
        i += 1;
    }
    &headers[start..i]
}

fn parse_content_length(headers: &[u8]) -> Option<usize> {
    let mut i = 0usize;
    while i + 15 <= headers.len() {
        if headers[i..i + 15].eq_ignore_ascii_case(b"content-length:") {
            let mut p = i + 15;
            while p < headers.len() && headers[p].is_ascii_whitespace() {
                p += 1;
            }
            let mut v = 0usize;
            while p < headers.len() && headers[p].is_ascii_digit() {
                v = v * 10 + (headers[p] - b'0') as usize;
                p += 1;
            }
            return Some(v);
        }
        i += 1;
    }
    None
}

fn is_keepalive(headers: &[u8]) -> bool {
    let mut i = 0usize;
    while i + 10 <= headers.len() {
        if headers[i..i + 10].eq_ignore_ascii_case(b"connection:") {
            let mut p = i + 10;
            while p < headers.len() && headers[p].is_ascii_whitespace() {
                p += 1;
            }
            let start = p;
            while p < headers.len() && !headers[p].is_ascii_whitespace() && headers[p] != b'\r' {
                p += 1;
            }
            let val = &headers[start..p];
            return val.eq_ignore_ascii_case(b"keep-alive");
        }
        i += 1;
    }
    true // HTTP/1.1 defaults to keep-alive
}

pub const HTTP_FRAUD: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

pub const RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
pub const RESP_NOT_FOUND: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
pub const RESP_BAD: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
pub const RESP_NOT_READY: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
