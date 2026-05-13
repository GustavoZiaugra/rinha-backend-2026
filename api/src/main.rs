mod data;
mod http;
mod json;
mod json_utils;
mod knn;
mod vector;

use mimalloc::MiMalloc;
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static READY: AtomicBool = AtomicBool::new(false);

pub fn is_ready() -> bool {
    READY.load(Ordering::Relaxed)
}

fn main() {
    let bind_addr = std::env::var("BIND")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    let listener = TcpListener::bind(&bind_addr).expect("bind TCP");

    // Initialize dataset and warmup in background so we can start serving right away
    std::thread::spawn(move || {
        data::init();
        knn::warmup();
        READY.store(true, Ordering::Relaxed);
    });

    http::serve(listener).expect("serve failed");
}
