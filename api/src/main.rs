mod data;
mod http;
mod json;
mod json_utils;
mod knn;
mod vector;

use libc;
use mimalloc::MiMalloc;
use std::net::TcpListener;
use std::os::unix::io::FromRawFd;
use std::sync::atomic::{AtomicBool, Ordering};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static READY: AtomicBool = AtomicBool::new(false);

pub fn is_ready() -> bool {
    READY.load(Ordering::Relaxed)
}

fn main() {
    // Create a TCP socket with a large listen backlog (16384)
    // to handle many concurrent connections without rejections.
    let listener = unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0);
        if fd < 0 {
            panic!("socket failed");
        }
        let opt: i32 = 1;
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_REUSEADDR,
            &opt as *const _ as *const libc::c_void,
            std::mem::size_of::<i32>() as libc::socklen_t,
        );
        let addr: libc::sockaddr_in = {
            let mut a: libc::sockaddr_in = std::mem::zeroed();
            a.sin_family = libc::AF_INET as libc::sa_family_t;
            a.sin_port = 8080u16.to_be();
            a.sin_addr = libc::in_addr {
                s_addr: 0u32, // INADDR_ANY
            };
            a
        };
        let ret = libc::bind(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
        );
        if ret < 0 {
            panic!("bind failed: {}", std::io::Error::last_os_error());
        }
        libc::listen(fd, 16384);
        TcpListener::from_raw_fd(fd)
    };

    // Initialize dataset and mark ready immediately
    std::thread::spawn(move || {
        data::init();
        READY.store(true, Ordering::Relaxed);
    });

    http::serve(listener).expect("serve failed");
}
