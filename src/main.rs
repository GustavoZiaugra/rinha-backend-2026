mod data;
mod http;
mod json;
mod knn;
mod vector;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    data::init();
    knn::warmup();

    let sock_path = std::env::var("SOCK")
        .unwrap_or_else(|_| "/run/sock/api.sock".to_string());

    http::serve(&sock_path).expect("serve failed");
}
