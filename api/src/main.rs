mod data;
mod http;
mod json;
mod knn;
mod vector;

use std::fs;
use std::os::unix::io::FromRawFd;
use std::sync::atomic::{AtomicBool, Ordering};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static READY: AtomicBool = AtomicBool::new(false);

pub fn is_ready() -> bool {
    READY.load(Ordering::Relaxed)
}

fn main() {
    // If ACCURACY_TEST env var is set, run accuracy test and exit
    if let Ok(nprobe_str) = std::env::var("ACCURACY_TEST") {
        let nprobe: usize = nprobe_str.parse().unwrap_or(15);
        let sample: usize = std::env::var("ACCURACY_SAMPLE")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(200);
        data::init();
        let (total, errors, fp, fn_) = knn::accuracy_test(nprobe, sample);
        if errors > 0 {
            eprintln!("ACCURACY FAIL: NPROBE={} | {}/{} errors (FP={}, FN={})",
                nprobe, errors, total, fp, fn_);
            std::process::exit(1);
        } else {
            eprintln!("ACCURACY PASS: NPROBE={} | {} queries, zero errors", nprobe, total);
            std::process::exit(0);
        }
    }

    let sock_path = std::env::var("SOCK").unwrap_or_else(|_| "/run/api.sock".to_string());

    // Remove stale socket file if present
    let _ = fs::remove_file(&sock_path);

    // Create a Unix Domain Socket with a large listen backlog
    let listener = unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0);
        if fd < 0 {
            panic!("socket failed: {}", std::io::Error::last_os_error());
        }

        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;

        // Copy path into sun_path (null-terminated) via raw pointer
        let bytes = sock_path.as_bytes();
        let len = bytes.len().min(107);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                addr.sun_path.as_mut_ptr() as *mut u8,
                len,
            );
            *addr.sun_path.as_mut_ptr().add(len) = 0;
        }

        let addr_len = std::mem::size_of::<libc::sa_family_t>() + len + 1;
        let ret = libc::bind(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            addr_len as libc::socklen_t,
        );
        if ret < 0 {
            panic!("bind failed: {}", std::io::Error::last_os_error());
        }

        libc::listen(fd, 16384);

        // Set permissions so HAProxy can connect
        libc::chmod(sock_path.as_ptr() as *const libc::c_char, 0o777);

        std::os::unix::net::UnixListener::from_raw_fd(fd)
    };

    // Initialize dataset in background, then warm up
    std::thread::spawn(move || {
        data::init();
        // Warm up: force page residency and prime CPU caches
        knn::warmup();
        READY.store(true, Ordering::Relaxed);
    });

    http::serve(listener).expect("serve failed");
}
