//! UDP auto-discovery beacon so 1bit Mobile finds this jarvis server on the
//! LAN without the user typing in an IP address. Port of jarvis/beacon.py --
//! matches the `BeaconListenerService` on port 13305 the app already listens
//! on (`{"service": "1bit", "hostname": ..., "url": ...}` broadcast datagrams).

use serde_json::json;
use tokio::net::UdpSocket;
use tokio::time::{sleep, Duration};

pub const BEACON_PORT: u16 = 13305;

fn lan_ip() -> String {
    match std::net::UdpSocket::bind("0.0.0.0:0").and_then(|s| {
        s.connect("8.8.8.8:80")?;
        s.local_addr()
    }) {
        Ok(addr) => addr.ip().to_string(),
        Err(_) => "127.0.0.1".to_string(),
    }
}

/// Starts a background task broadcasting this server's address every
/// `interval` seconds. Returns immediately; runs for the life of the process.
pub async fn start_beacon(port: u16, interval: Duration) {
    let url = format!("http://{}:{port}", lan_ip());
    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "jarvis".to_string());
    let payload = json!({ "service": "1bit", "hostname": hostname, "url": url }).to_string();

    println!("  Beacon: advertising {url} on UDP :{BEACON_PORT}");

    tokio::spawn(async move {
        let socket = match UdpSocket::bind("0.0.0.0:0").await {
            Ok(s) => s,
            Err(_) => return,
        };
        if socket.set_broadcast(true).is_err() {
            return;
        }
        loop {
            let _ = socket.send_to(payload.as_bytes(), ("255.255.255.255", BEACON_PORT)).await;
            sleep(interval).await;
        }
    });
}
