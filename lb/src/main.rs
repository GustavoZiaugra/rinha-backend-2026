use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io;
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Builder;

fn main() {
    let rt = Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("runtime");

    rt.block_on(async {
        let port: u16 = std::env::var("PORT")
            .unwrap_or_else(|_| "9999".into())
            .parse()
            .expect("invalid PORT");
        let upstreams: Vec<String> = std::env::var("UPSTREAMS")
            .unwrap_or_else(|_| "api1:8080,api2:8080".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
        let listener = TcpListener::bind(addr).await.expect("bind");
        eprintln!("lb listening on {addr}");

        let upstreams = Arc::new(upstreams);
        let counter = Arc::new(AtomicUsize::new(0));

        loop {
            let (mut client, _) = listener.accept().await.expect("accept");
            client.set_nodelay(true).ok();

            let upstreams = upstreams.clone();
            let counter = counter.clone();

            tokio::spawn(async move {
                let idx = counter.fetch_add(1, Ordering::Relaxed) % upstreams.len();
                let backend = &upstreams[idx];

                match TcpStream::connect(backend).await {
                    Ok(mut server) => {
                        server.set_nodelay(true).ok();
                        let _ = io::copy_bidirectional(&mut client, &mut server).await;
                    }
                    Err(_) => {
                        let resp = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
                        let _ = io::AsyncWriteExt::write_all(&mut client, resp).await;
                    }
                }
            });
        }
    });
}
