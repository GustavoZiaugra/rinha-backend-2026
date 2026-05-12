mod data;
mod http;
mod json;
mod knn;
mod vector;

use mimalloc::MiMalloc;
use std::os::unix::net::UnixListener;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static READY: AtomicBool = AtomicBool::new(false);

pub fn is_ready() -> bool {
    READY.load(Ordering::Relaxed)
}

fn main() {
    let sock_path = std::env::var("SOCK")
        .unwrap_or_else(|_| "/run/sock/api.sock".to_string());

    // Create socket IMMEDIATELY so the LB can connect
    let path = PathBuf::from(&sock_path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind UDS");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666))
        .expect("set perms");

    // Initialize dataset and warmup in background so we can start serving right away
    std::thread::spawn(move || {
        data::init();
        knn::warmup();
        READY.store(true, Ordering::Relaxed);
    });

    http::serve(listener).expect("serve failed");
}
