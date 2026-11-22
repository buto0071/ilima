use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpStream, UdpSocket};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use futures::StreamExt;
use crate::cluster_config::ClusterConfig;

/// Pinned llama.cpp release tag used when auto-bootstrapping a fresh node.
pub const PINNED_LLAMA_TAG: &str = "b12503";

/// Returns all non-loopback IPv4 addresses assigned to active network interfaces on this host.
/// Filters out loopback (`127.x.x.x`) and link-local (`169.254.x.x`) addresses.
pub fn detect_local_ipv4s() -> Vec<String> {
    let mut ips: Vec<String> = Vec::new();

    if let Ok(ifaces) = local_ip_address::list_afinet_netifas() {
        for (_name, ip) in ifaces {
            if let IpAddr::V4(v4) = ip {
                let octets = v4.octets();
                // Skip loopback (127.x.x.x) and APIPA link-local (169.254.x.x)
                if !v4.is_loopback() && !(octets[0] == 169 && octets[1] == 254) {
                    let s = v4.to_string();
                    if !ips.contains(&s) {
                        ips.push(s);
                    }
                }
            }
        }
    }

    // Fallback via UDP routing table probe if list_afinet_netifas returned nothing
    if ips.is_empty() {
        if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
            if sock.connect("8.8.8.8:80").is_ok() {
                if let Ok(local_addr) = sock.local_addr() {
                    if let IpAddr::V4(v4) = local_addr.ip() {
                        if !v4.is_loopback() {
                            ips.push(v4.to_string());
                        }
                    }
                }
            }
        }
    }

    // Sort physical LAN IPs (192.168.x.x, 10.x.x.x) ahead of virtual/VPN interfaces (e.g. Tailscale 100.64.0.0/10 or WSL 172.16.0.0/12)
    ips.sort_by_key(|ip| {
        if ip.starts_with("192.168.") {
            0
        } else if ip.starts_with("10.") {
            1
        } else {
            2
        }
    });

    ips
}

/// Checks if a local TCP port is already bound and accepting connections.
pub fn is_local_port_open(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

/// Ensures `cfg.llama_bin_dir` points to a valid directory containing `llama-server` or `rpc-server`.
/// If the configured path does not exist on this machine, scans common portable locations.
pub fn ensure_runtime_dir(cfg: &mut ClusterConfig) -> bool {
    if cfg.use_wsl {
        return true;
    }
    let server_bin = cfg.binary("llama-server");
    if Path::new(&server_bin).exists() {
        return true;
    }

    let exe_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let candidate_dirs = [
        "./runtime",
        "./runtime/build/bin",
        "./bin",
        "./llama.cpp",
        "../llama-b12503-bin-win-vulkan-x64",
        "../llama-b12038-bin-win-vulkan-x64",
        "C:\\llama.cpp",
    ];

    for dir in candidate_dirs {
        let candidate = Path::new(dir).join(exe_name);
        if candidate.exists() {
            println!("Auto-detected llama.cpp runtime in {}", dir);
            cfg.llama_bin_dir = dir.to_string();
            return true;
        }
    }
    false
}

/// Returns the official GitHub release download URL for the current OS/architecture.
pub fn default_runtime_download_url() -> String {
    if cfg!(target_os = "windows") {
        format!(
            "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-win-vulkan-x64.zip",
            tag = PINNED_LLAMA_TAG
        )
    } else if cfg!(target_os = "macos") {
        format!(
            "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-macos-arm64.zip",
            tag = PINNED_LLAMA_TAG
        )
    } else {
        format!(
            "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-vulkan-x64.zip",
            tag = PINNED_LLAMA_TAG
        )
    }
}

/// Checks if the llama.cpp runtime exists locally; if not, automatically downloads and extracts
/// the official prebuilt Vulkan/Metal release into `./runtime`.
pub async fn ensure_runtime_ready(cfg: &mut ClusterConfig) {
    if ensure_runtime_dir(cfg) {
        return;
    }

    let url = default_runtime_download_url();
    let runtime_dir = Path::new("./runtime");
    let _ = std::fs::create_dir_all(runtime_dir);
    let zip_path = runtime_dir.join("llama-runtime.zip");

    println!(
        "llama-server not found at '{}'. Auto-downloading portable runtime from {}...",
        cfg.llama_bin_dir, url
    );

    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(file) = std::fs::File::create(&zip_path) {
                let mut writer = std::io::BufWriter::new(file);
                let mut stream = resp.bytes_stream();
                while let Some(Ok(chunk)) = stream.next().await {
                    let _ = writer.write_all(&chunk);
                }
                let _ = writer.flush();
            }

            let zip_str = zip_path.to_string_lossy().to_string();
            // Extract using system `tar -xf` (available on Windows 10/11, Linux, and macOS)
            let status = Command::new("tar")
                .args(["-xf", &zip_str, "-C", "./runtime"])
                .status();

            if status.map(|s| !s.success()).unwrap_or(true) && cfg!(windows) {
                let _ = Command::new("powershell")
                    .args([
                        "-NoProfile",
                        "-Command",
                        &format!(
                            "Expand-Archive -Path '{}' -DestinationPath './runtime' -Force",
                            zip_str
                        ),
                    ])
                    .status();
            }
            let _ = std::fs::remove_file(&zip_path);
            let _ = ensure_runtime_dir(cfg);
        }
        Ok(resp) => {
            eprintln!(
                "Warning: Failed to auto-download runtime (HTTP {}). Please set llama_bin_dir in ilima.json.",
                resp.status()
            );
        }
        Err(e) => {
            eprintln!(
                "Warning: Could not download runtime ({}). Please set llama_bin_dir in ilima.json.",
                e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_local_ipv4s_excludes_loopback() {
        let ips = detect_local_ipv4s();
        for ip in &ips {
            assert!(!ip.starts_with("127."));
            assert!(!ip.starts_with("169.254."));
        }
    }

    #[test]
    fn test_default_runtime_download_url_is_valid() {
        let url = default_runtime_download_url();
        assert!(url.starts_with("https://github.com/ggml-org/llama.cpp/releases/download/"));
        assert!(url.ends_with(".zip"));
    }
}