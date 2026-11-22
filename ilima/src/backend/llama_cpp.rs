use std::collections::HashSet;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tracing::info;

use crate::cluster_config::ClusterConfig;
use crate::scheduler::{DiscoveredWorker, PeerInfo};

/// Result of the N-Node Greedy Placement & Split Engine.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementPlan {
    /// Ordered list of RPC worker endpoints (`ip:port`) recruited for this model.
    /// Empty if the model fits 100% inside the Queen's local VRAM.
    pub selected_workers: Vec<String>,
    /// Value for `--tensor-split` matching `[worker_1, ..., worker_{k-1}, queen]`.
    pub tensor_split: Option<String>,
}

pub struct LlamaCppBackend;

impl LlamaCppBackend {
    /// Runs the setup script in WSL to build llama.cpp (only needed when use_wsl = true).
    pub fn setup() -> Result<(), String> {
        info!("Setting up llama.cpp in WSL...");

        let output = Command::new("wsl")
            .arg("find")
            .arg(".")
            .arg("-name")
            .arg("setup_llama.sh")
            .output()
            .map_err(|e| format!("Failed to run find command: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let found_path = stdout
            .lines()
            .next()
            .ok_or("Could not find scripts/setup_llama.sh in current directory or subdirectories.")?
            .trim()
            .to_string();

        info!("Found setup script at: {}", found_path);

        let _ = Command::new("wsl")
            .arg("bash")
            .arg("-c")
            .arg(format!(
                "tr -d '\\r' < {} > {}.tmp && mv {}.tmp {}",
                found_path, found_path, found_path, found_path
            ))
            .status();

        let status = Command::new("wsl")
            .arg("bash")
            .arg(&found_path)
            .status()
            .map_err(|e| format!("Failed to execute wsl command: {}", e))?;

        if status.success() {
            info!("llama.cpp setup complete.");
            Ok(())
        } else {
            Err(format!("Setup script failed with status: {}", status))
        }
    }

    /// Convert a Windows path to a WSL path when use_wsl = true. Otherwise return it unchanged.
    fn to_backend_path(cfg: &ClusterConfig, path: &str) -> String {
        if !cfg.use_wsl {
            return path.to_string();
        }
        match Command::new("wsl").arg("wslpath").arg("-a").arg(path).output() {
            Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
            Err(_) => path.to_string(),
        }
    }

    /// Build a Command for a llama.cpp binary. No shell is used, so no quoting problems.
    fn build_command(cfg: &ClusterConfig, binary_name: &str, args: &[String]) -> Command {
        let bin = cfg.binary(binary_name);
        if cfg.use_wsl {
            let mut cmd = Command::new("wsl");
            cmd.arg("--").arg(bin).args(args);
            cmd
        } else {
            let mut cmd = Command::new(bin);
            cmd.args(args);
            cmd
        }
    }

    /// Arguments for rpc-server. `-c` turns on the local weight cache.
    pub fn worker_args(port: u16) -> Vec<String> {
        vec![
            "-H".to_string(),
            "0.0.0.0".to_string(),
            "-p".to_string(),
            port.to_string(),
            "-c".to_string(),
        ]
    }

    /// Starts the RPC Worker (rpc-server). Blocks until rpc-server exits.
    /// Automatically detects the best discrete GPU device (e.g. Vulkan0 / Vulkan1 / CUDA0)
    /// and passes `-d <device>` so rpc-server never falls back to CPU or an integrated GPU.
    pub fn start_worker(cfg: &ClusterConfig, port: u16) -> Result<(), String> {
        let mut args = Self::worker_args(port);
        if let Some(dev_id) = Self::probe_local_best_device_id(cfg) {
            info!("Binding rpc-server to discrete GPU device: {}", dev_id);
            args.push("-d".to_string());
            args.push(dev_id);
        }
        info!("Starting rpc-server: {} {}", cfg.binary("rpc-server"), args.join(" "));
        let status = Self::build_command(cfg, "rpc-server", &args)
            .status()
            .map_err(|e| format!("Failed to start rpc-server: {}", e))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("rpc-server exited with status: {}", status))
        }
    }

    /// Arguments for llama-server, built from the cluster config.
    /// If the Queen has multiple local GPUs (e.g. an iGPU `Vulkan0` + discrete `Vulkan1`) and the user
    /// did not specify `-dev` in `extra_server_args`, automatically pins `-dev` to the RPC devices +
    /// the Queen's best discrete GPU so `--tensor-split` never accidentally lands on an integrated GPU.
    pub fn server_args(cfg: &ClusterConfig, model_path: &str, workers: &[String]) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "-m".to_string(),
            Self::to_backend_path(cfg, model_path),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            cfg.server_port.to_string(),
            "-ngl".to_string(),
            cfg.ngl.to_string(),
            "-c".to_string(),
            cfg.ctx_size.to_string(),
            "-np".to_string(),
            cfg.parallel_slots.to_string(),
        ];
        if cfg.cache_reuse > 0 {
            args.push("--cache-reuse".to_string());
            args.push(cfg.cache_reuse.to_string());
        }
        if !workers.is_empty() {
            args.push("--rpc".to_string());
            args.push(workers.join(","));
        }
        // If Queen has multiple local GPUs (e.g. iGPU + dGPU), explicitly bind -dev so the iGPU
        // does not steal a --tensor-split slot.
        let has_manual_dev = cfg
            .extra_server_args
            .iter()
            .any(|a| a == "-dev" || a == "--device");
        if !has_manual_dev {
            let (local_gpu_count, best_local_dev) = Self::probe_local_gpu_count_and_best_id(cfg);
            if local_gpu_count > 1 {
                if let Some(local_dev) = best_local_dev {
                    let mut dev_list: Vec<String> = workers
                        .iter()
                        .map(|w| format!("RPC[{}]", w))
                        .collect();
                    dev_list.push(local_dev);
                    args.push("-dev".to_string());
                    args.push(dev_list.join(","));
                }
            }
        }
        if let Some(ts) = &cfg.tensor_split {
            args.push("--tensor-split".to_string());
            args.push(ts.clone());
        }
        if let Some(n) = cfg.n_cpu_moe {
            args.push("--n-cpu-moe".to_string());
            args.push(n.to_string());
        }
        if let Some(st) = &cfg.spec_type {
            args.push("--spec-type".to_string());
            args.push(st.clone());
            let has_ctx_cp = cfg
                .extra_server_args
                .iter()
                .any(|a| a == "--ctx-checkpoints" || a == "-ctxcp");
            if !has_ctx_cp {
                args.push("--ctx-checkpoints".to_string());
                args.push("0".to_string());
            }
        }
        if let Some(draft) = &cfg.draft_model {
            args.push("-md".to_string());
            args.push(Self::to_backend_path(cfg, draft));
            args.push("-ngld".to_string());
            args.push("99".to_string());
            if cfg.draft_max > 0 {
                args.push("--spec-draft-n-max".to_string());
                args.push(cfg.draft_max.to_string());
            }
            if cfg.draft_min > 0 {
                args.push("--spec-draft-n-min".to_string());
                args.push(cfg.draft_min.to_string());
            }
            if let Some(dev) = &cfg.draft_device {
                args.push("-devd".to_string());
                args.push(dev.clone());
            }
        } else if cfg.spec_type.is_some() && cfg.draft_max > 0 {
            args.push("--spec-draft-n-max".to_string());
            args.push(cfg.draft_max.to_string());
            if cfg.draft_min > 0 {
                args.push("--spec-draft-n-min".to_string());
                args.push(cfg.draft_min.to_string());
            }
        }
        if let Some(re) = &cfg.reasoning_effort {
            args.push("--reasoning-effort".to_string());
            args.push(re.clone());
        }
        if let Some(rb) = cfg.reasoning_budget {
            args.push("--reasoning-budget".to_string());
            args.push(rb.to_string());
        }
        args.extend(cfg.extra_server_args.iter().cloned());
        args
    }

    /// Kill any stale llama-server still holding `cfg.server_port` (without killing unrelated
    /// llama-server instances on other ports, such as `llama-proxy.py` on port 9090).
    pub fn kill_stale_servers(cfg: &ClusterConfig) {
        if !crate::runtime_manager::is_local_port_open(cfg.server_port) {
            return;
        }
        if cfg.use_wsl {
            let _ = Command::new("wsl")
                .args([
                    "fuser",
                    "-k",
                    &format!("{}/tcp", cfg.server_port),
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        } else if cfg!(windows) {
            let ps_cmd = format!(
                "Get-NetTCPConnection -LocalPort {} -State Listen -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}",
                cfg.server_port
            );
            let _ = Command::new("powershell")
                .args(["-NoProfile", "-Command", &ps_cmd])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        } else {
            let _ = Command::new("fuser")
                .args(["-k", &format!("{}/tcp", cfg.server_port)])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }

    /// Starts llama-server in the background and returns the child process.
    pub fn start_server(cfg: &ClusterConfig, model_path: &str, workers: &[String]) -> Result<Child, String> {
        let args = Self::server_args(cfg, model_path, workers);
        info!("Starting llama-server: {} {}", cfg.binary("llama-server"), args.join(" "));
        Self::build_command(cfg, "llama-server", &args)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn llama-server: {}", e))
    }

    /// Parses `llama-server --list-devices` output and returns `(total_mb, free_mb, gpu_name)`
    /// for the best local non-RPC GPU (preferring discrete GPUs with the largest free VRAM over iGPUs).
    pub fn parse_best_gpu_from_device_list(combined_output: &str) -> (u64, u64, Option<String>) {
        let (total_mb, free_mb, clean_name, _dev_id) =
            Self::parse_best_gpu_with_id_from_device_list(combined_output);
        (total_mb, free_mb, clean_name)
    }

    /// Parses `llama-server --list-devices` output and returns `(total_mb, free_mb, gpu_name, device_id)`
    /// where `device_id` is e.g. `"Vulkan0"`, `"Vulkan1"`, or `"CUDA0"`.
    pub fn parse_best_gpu_with_id_from_device_list(
        combined_output: &str,
    ) -> (u64, u64, Option<String>, Option<String>) {
        let mut best: (u64, u64, Option<String>, Option<String>) = (0, 0, None, None);

        for line in combined_output.lines() {
            if line.contains("MiB free)") && !line.contains("RPC") && !line.contains("CPU") {
                let raw_name = line.split('(').next().map(|s| s.trim().to_string());
                let (dev_id, clean_name) = match raw_name {
                    Some(n) => {
                        if let Some((prefix, rest)) = n.split_once(':') {
                            (
                                Some(prefix.trim().to_string()),
                                Some(rest.trim().to_string()),
                            )
                        } else {
                            (None, Some(n))
                        }
                    }
                    None => (None, None),
                };
                if let Some(inside) = line.split('(').nth(1) {
                    // Format: "16384 MiB, 15800 MiB free)"
                    let nums: Vec<u64> = inside
                        .split(|c: char| !c.is_ascii_digit())
                        .filter(|s| !s.is_empty())
                        .filter_map(|s| s.parse::<u64>().ok())
                        .collect();
                    if nums.len() >= 2 {
                        let total_mb = nums[0];
                        let free_mb = nums[1];
                        let bw_mult = Self::gpu_bandwidth_multiplier(clean_name.as_deref());
                        let best_mult = Self::gpu_bandwidth_multiplier(best.2.as_deref());
                        let weighted_free = (free_mb as f64 * bw_mult) as u64;
                        let best_weighted = (best.1 as f64 * best_mult) as u64;
                        if weighted_free > best_weighted {
                            best = (total_mb, free_mb, clean_name, dev_id);
                        }
                    }
                }
            }
        }
        best
    }

    /// Runs a short-lived probe command (like `--list-devices`) with a strict wall-clock timeout,
    /// killing the child process if it hangs so a stalled RPC host or GPU driver never blocks the Queen.
    fn run_command_with_timeout(mut cmd: Command, timeout: Duration) -> Option<String> {
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(_)) => {
                    let out = child.wait_with_output().ok()?;
                    return Some(format!(
                        "{}\n{}",
                        String::from_utf8_lossy(&out.stdout),
                        String::from_utf8_lossy(&out.stderr)
                    ));
                }
                Ok(None) => {
                    if start.elapsed() >= timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
            }
        }
    }

    /// Returns `(non_rpc_gpu_count, best_device_id)` from `llama-server --list-devices`.
    pub fn probe_local_gpu_count_and_best_id(cfg: &ClusterConfig) -> (usize, Option<String>) {
        let args = vec!["--list-devices".to_string()];
        let Some(combined) = Self::run_command_with_timeout(
            Self::build_command(cfg, "llama-server", &args),
            Duration::from_secs(4),
        ) else {
            return (0, None);
        };
        let count = combined
            .lines()
            .filter(|l| l.contains("MiB free)") && !l.contains("RPC") && !l.contains("CPU"))
            .count();
        let best_id = Self::parse_best_gpu_with_id_from_device_list(&combined).3;
        (count, best_id)
    }

    /// Parses RPC device lines from `llama-server --rpc <endpoint> --list-devices` output.
    /// Matches lines like: `RPC0[192.168.1.50:50052]: Vulkan0 (8192 MiB, 7070 MiB free)`
    /// or `RPC[192.168.1.50:50052]: NVIDIA GeForce RTX 5050 (8150 MiB, 7070 MiB free)`.
    pub fn parse_rpc_device_from_list(combined_output: &str) -> Option<(u64, u64, Option<String>)> {
        for line in combined_output.lines() {
            if line.contains("RPC") && line.contains("MiB free)") {
                let raw_prefix = line.split('(').next().unwrap_or("").trim();
                let dev_name = raw_prefix
                    .rsplit_once(':')
                    .map(|(_, rest)| rest.trim().to_string())
                    .filter(|s| !s.is_empty() && !s.chars().all(|c| c.is_ascii_digit()));
                if let Some(inside) = line.split('(').nth(1) {
                    let nums: Vec<u64> = inside
                        .split(|c: char| !c.is_ascii_digit())
                        .filter(|s| !s.is_empty())
                        .filter_map(|s| s.parse::<u64>().ok())
                        .collect();
                    if nums.len() >= 2 {
                        return Some((nums[0], nums[1], dev_name));
                    }
                }
            }
        }
        None
    }

    /// Queries a remote RPC worker via `llama-server --rpc <endpoint> --list-devices`
    /// to get its exact `(total_vram_mb, free_vram_mb, device_name)` and verify RPC protocol compatibility.
    pub fn probe_rpc_worker_vram_mb(
        cfg: &ClusterConfig,
        endpoint: &str,
    ) -> Option<(u64, u64, Option<String>)> {
        let args = vec![
            "--rpc".to_string(),
            endpoint.to_string(),
            "--list-devices".to_string(),
        ];
        let combined = Self::run_command_with_timeout(
            Self::build_command(cfg, "llama-server", &args),
            Duration::from_secs(4),
        )?;
        Self::parse_rpc_device_from_list(&combined)
    }

    /// Looks up the Layer-2 MAC address for `ip` in the OS ARP table (`arp -a`).
    /// Used to distinguish two interfaces on the same multi-homed laptop vs. two separate machines
    /// that happen to have identical GPU models.
    pub fn lookup_arp_mac(ip: &str) -> Option<String> {
        let out = Command::new("arp").arg("-a").output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.iter().any(|t| *t == ip || t.trim_matches(|c| c == '(' || c == ')') == ip) {
                for tok in &tokens {
                    let clean = tok.replace('-', ":").to_ascii_lowercase();
                    if clean.len() == 17 && clean.chars().filter(|&c| c == ':').count() == 5 {
                        return Some(clean);
                    }
                }
            }
        }
        None
    }

    /// Returns true if `ip` is on a physical home/office LAN subnet (`192.168.x.x` or `10.x.x.x`),
    /// skipping `172.16.0.0/12` virtual bridges (WSL2, Hyper-V, Docker) so `/24` sweeps never scan virtual adapters.
    pub fn is_physical_lan_ipv4(ip: &str) -> bool {
        let parts: Vec<u8> = ip
            .split('.')
            .filter_map(|s| s.parse::<u8>().ok())
            .collect();
        if parts.len() != 4 {
            return false;
        }
        parts[0] == 192 && parts[1] == 168 || parts[0] == 10
    }

    /// Returns the best local discrete GPU device identifier (e.g. `"Vulkan0"` or `"Vulkan1"`).
    pub fn probe_local_best_device_id(cfg: &ClusterConfig) -> Option<String> {
        let args = vec!["--list-devices".to_string()];
        let combined = Self::run_command_with_timeout(
            Self::build_command(cfg, "llama-server", &args),
            Duration::from_secs(4),
        )?;
        Self::parse_best_gpu_with_id_from_device_list(&combined).3
    }

    /// Bandwidth tier multiplier based on GPU model name so discrete GPUs outrank integrated GPUs.
    pub fn gpu_bandwidth_multiplier(gpu_name: Option<&str>) -> f64 {
        let Some(name) = gpu_name else {
            return 1.0;
        };
        let lower = name.to_ascii_lowercase();
        // Penalize integrated GPUs sharing system DDR4/DDR5 RAM
        if lower.contains("uhd graphics")
            || lower.contains("iris")
            || lower == "amd radeon(tm) graphics"
            || lower.contains("radeon(tm) graphics")
            || lower.contains("llvmpipe")
        {
            return 0.25;
        }
        // Boost high-bandwidth desktop/workstation GPUs
        if lower.contains("9070")
            || lower.contains("7900")
            || lower.contains("4090")
            || lower.contains("4080")
            || lower.contains("5090")
            || lower.contains("5080")
        {
            return 1.25;
        }
        1.0
    }

    /// Probe local GPU free VRAM (in MB) using `llama-server --list-devices` first,
    /// falling back to `cfg.local_vram_mb` only if `--list-devices` cannot be queried.
    pub fn probe_local_vram_mb(cfg: &ClusterConfig) -> (u64, u64, Option<String>) {
        let args = vec!["--list-devices".to_string()];
        if let Some(combined) = Self::run_command_with_timeout(
            Self::build_command(cfg, "llama-server", &args),
            Duration::from_secs(4),
        ) {
            let (total, free, name) = Self::parse_best_gpu_from_device_list(&combined);
            if total > 0 {
                return (total, free, name);
            }
        }
        if cfg.local_vram_mb > 0 {
            return (cfg.local_vram_mb, cfg.local_vram_mb, None);
        }
        (0, 0, None)
    }

    /// Measure TCP handshake round-trip time (RTT) in milliseconds to `endpoint` (`ip:port`).
    pub async fn measure_tcp_rtt_ms(endpoint: &str, timeout_ms: u64) -> Option<f64> {
        let start = Instant::now();
        let fut = tokio::net::TcpStream::connect(endpoint);
        match tokio::time::timeout(Duration::from_millis(timeout_ms), fut).await {
            Ok(Ok(_stream)) => Some(start.elapsed().as_secs_f64() * 1000.0),
            _ => None,
        }
    }

    /// Fast parallel `/24` LAN sweep for standalone `rpc-server` instances on `rpc_port`.
    /// Only scans physical LAN subnets (`192.168.x.x` and `10.x.x.x`), skipping WSL/Hyper-V/Docker bridges.
    pub async fn sweep_subnet_for_rpc(local_ips: &[String], rpc_port: u16) -> Vec<(String, f64)> {
        let mut prefixes: Vec<String> = Vec::new();
        for ip in local_ips {
            if !Self::is_physical_lan_ipv4(ip) {
                continue;
            }
            let parts: Vec<&str> = ip.split('.').collect();
            if parts.len() == 4 {
                let prefix = format!("{}.{}.{}", parts[0], parts[1], parts[2]);
                if !prefixes.contains(&prefix) {
                    prefixes.push(prefix);
                }
            }
        }

        let local_set: HashSet<String> = local_ips.iter().cloned().collect();
        let mut handles = Vec::new();

        for prefix in prefixes {
            for host in 1..=254u8 {
                let candidate_ip = format!("{}.{}", prefix, host);
                if local_set.contains(&candidate_ip) {
                    continue;
                }
                let endpoint = format!("{}:{}", candidate_ip, rpc_port);
                handles.push(tokio::spawn(async move {
                    Self::measure_tcp_rtt_ms(&endpoint, 450)
                        .await
                        .map(|rtt| (endpoint, rtt))
                }));
            }
        }

        let mut found = Vec::new();
        for h in handles {
            if let Ok(Some(item)) = h.await {
                found.push(item);
            }
        }
        found
    }

    /// Autonomous N-Node Worker Discovery & Multi-Interface RTT Ranker.
    ///
    /// 1. Examines every discovered `libp2p` peer's `Multiaddr` IPs and `lan_ips` from heartbeats.
    /// 2. Benchmarks TCP RTT to all interfaces of each peer and selects the **fastest reachable IP**
    ///    per peer (e.g. Ethernet `0.3ms` over Wi-Fi `8.0ms`).
    /// 3. Also probes any static fallback entries in `cfg.workers`.
    /// 4. Sweeps local `/24` subnet when `cfg.workers` is empty to discover standalone `rpc-server`
    ///    instances, probing their real VRAM via `--list-devices` and deduplicating multi-homed interfaces.
    /// 5. Ranks all reachable workers by effective speed score (low RTT + high free VRAM first).
    pub async fn discover_and_rank_workers(
        cfg: &ClusterConfig,
        peers: &[PeerInfo],
        local_ips: &[String],
    ) -> Vec<DiscoveredWorker> {
        let mut local_set: HashSet<String> = local_ips.iter().cloned().collect();
        local_set.insert("127.0.0.1".to_string());
        local_set.insert("0.0.0.0".to_string());

        let mut discovered: Vec<DiscoveredWorker> = Vec::new();
        let mut seen_ips: HashSet<String> = HashSet::new();

        // 1. Evaluate libp2p peers (picking the lowest-RTT interface for each peer)
        for peer in peers {
            let port = if peer.capabilities.rpc_port > 0 {
                peer.capabilities.rpc_port
            } else {
                cfg.rpc_port
            };

            let mut candidate_ips: Vec<String> = Vec::new();
            for ip in &peer.capabilities.lan_ips {
                if !local_set.contains(ip) && !candidate_ips.contains(ip) {
                    candidate_ips.push(ip.clone());
                }
            }
            for addr in &peer.address {
                let s = addr.to_string();
                let parts: Vec<&str> = s.split('/').collect();
                if parts.len() >= 3 && parts[1] == "ip4" {
                    let ip = parts[2].to_string();
                    if !local_set.contains(&ip) && !candidate_ips.contains(&ip) {
                        candidate_ips.push(ip);
                    }
                }
            }

            let mut best_for_peer: Option<(String, f64)> = None;
            for ip in &candidate_ips {
                let endpoint = format!("{}:{}", ip, port);
                if let Some(rtt) = Self::measure_tcp_rtt_ms(&endpoint, 600).await {
                    match &best_for_peer {
                        Some((_, best_rtt)) if rtt >= *best_rtt => {}
                        _ => best_for_peer = Some((endpoint, rtt)),
                    }
                }
            }

            if let Some((best_endpoint, best_rtt)) = best_for_peer {
                for ip in candidate_ips {
                    seen_ips.insert(ip);
                }
                if let Some(ip_only) = best_endpoint.split(':').next() {
                    seen_ips.insert(ip_only.to_string());
                }
                discovered.push(DiscoveredWorker {
                    endpoint: best_endpoint,
                    rtt_ms: best_rtt,
                    total_vram_mb: peer.capabilities.total_vram_mb,
                    available_vram_mb: peer.capabilities.available_vram_mb,
                    gpu_name: peer.capabilities.gpu_name.clone(),
                });
            }
        }

        // 2. Evaluate any static workers listed in cfg.workers (if not already discovered above)
        for raw in &cfg.workers {
            let endpoint = if raw.contains(':') {
                raw.clone()
            } else {
                format!("{}:{}", raw, cfg.rpc_port)
            };
            let ip_only = endpoint.split(':').next().unwrap_or("").to_string();
            if local_set.contains(&ip_only) || seen_ips.contains(&ip_only) {
                continue;
            }
            if let Some(rtt) = Self::measure_tcp_rtt_ms(&endpoint, 600).await {
                seen_ips.insert(ip_only);
                let (total_vram_mb, available_vram_mb, gpu_name) =
                    Self::probe_rpc_worker_vram_mb(cfg, &endpoint).unwrap_or((0, 0, None));
                discovered.push(DiscoveredWorker {
                    endpoint,
                    rtt_ms: rtt,
                    total_vram_mb,
                    available_vram_mb,
                    gpu_name,
                });
            }
        }

        // 3. Also sweep local /24 subnet when cfg.workers is empty so any standalone `rpc-server`
        //    is discovered, queried for real VRAM via `--list-devices`, and deduplicated across
        //    multi-homed interfaces (e.g. Laptop connected via both Wi-Fi .142 and Ethernet .196).
        if cfg.workers.is_empty() {
            let mut swept = Self::sweep_subnet_for_rpc(local_ips, cfg.rpc_port).await;
            // Sort by lowest RTT first so Ethernet (<1ms) is evaluated before Wi-Fi (~8ms)
            swept.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut seen_macs: HashSet<String> = HashSet::new();
            let mut seen_eth_gpus: Vec<(u64, u64, Option<String>)> = discovered
                .iter()
                .filter(|w| w.total_vram_mb > 0)
                .map(|w| (w.total_vram_mb, w.available_vram_mb, w.gpu_name.clone()))
                .collect();

            for (endpoint, rtt_ms) in swept {
                let ip_only = endpoint.split(':').next().unwrap_or("").to_string();
                if seen_ips.contains(&ip_only) || local_set.contains(&ip_only) {
                    continue;
                }
                if let Some(mac) = Self::lookup_arp_mac(&ip_only) {
                    if !seen_macs.insert(mac) {
                        seen_ips.insert(ip_only);
                        continue;
                    }
                }

                let (total_vram_mb, available_vram_mb, gpu_name) =
                    Self::probe_rpc_worker_vram_mb(cfg, &endpoint).unwrap_or((0, 0, None));

                if total_vram_mb > 0 {
                    // If we already found a fast link (e.g. Ethernet) with the exact same GPU and nearly
                    // identical free VRAM (within 32 MB), and this second candidate is a slower Wi-Fi link
                    // (rtt_ms > 2.0), skip the redundant Wi-Fi interface of that multi-homed machine.
                    let is_dup_wifi = rtt_ms > 2.0
                        && seen_eth_gpus.iter().any(|(tot, free, name)| {
                            *tot == total_vram_mb
                                && *name == gpu_name
                                && (*free as i64 - available_vram_mb as i64).unsigned_abs() <= 32
                        });
                    if is_dup_wifi {
                        info!(
                            "Skipping duplicate Wi-Fi interface {} (RTT {:.2} ms) -- same GPU already discovered on faster Ethernet link.",
                            endpoint, rtt_ms
                        );
                        seen_ips.insert(ip_only);
                        continue;
                    }
                    seen_eth_gpus.push((total_vram_mb, available_vram_mb, gpu_name.clone()));
                }

                info!(
                    "Auto-discovered standalone RPC server at {} (RTT {:.2} ms, free VRAM {} MB)",
                    endpoint, rtt_ms, available_vram_mb
                );
                seen_ips.insert(ip_only);
                discovered.push(DiscoveredWorker {
                    endpoint,
                    rtt_ms,
                    total_vram_mb,
                    available_vram_mb,
                    gpu_name,
                });
            }
        }

        // 4. Sort workers by effective score: (VRAM * bandwidth_multiplier) / (1.0 + RTT_ms * 0.5)
        discovered.sort_by(|a, b| {
            let vram_a = a.available_vram_mb.max(6600) as f64
                * Self::gpu_bandwidth_multiplier(a.gpu_name.as_deref());
            let vram_b = b.available_vram_mb.max(6600) as f64
                * Self::gpu_bandwidth_multiplier(b.gpu_name.as_deref());
            let score_a = vram_a / (1.0 + a.rtt_ms * 0.5);
            let score_b = vram_b / (1.0 + b.rtt_ms * 0.5);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        discovered
    }

    /// Computes total model size in MB on disk, automatically summing all shards if `model_path`
    /// is a multi-part GGUF (e.g. `Model-00001-of-00003.gguf`) plus any configured `draft_model`.
    pub fn estimate_total_model_mb(cfg: &ClusterConfig, model_path: &str) -> u64 {
        let mut total_bytes: u64 = 0;
        let p = Path::new(model_path);

        if let (Some(parent), Some(fname)) = (p.parent(), p.file_name().and_then(|f| f.to_str())) {
            if let Some((prefix, rest)) = fname.rsplit_once("-00001-of-") {
                let suffix = rest.trim_start_matches(|c: char| c.is_ascii_digit());
                if let Ok(entries) = std::fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if let Some(en) = entry.file_name().to_str() {
                            if en.starts_with(prefix) && en.ends_with(suffix) && en.contains("-of-") {
                                total_bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
            }
        }

        if total_bytes == 0 {
            total_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
        }

        if let Some(draft) = &cfg.draft_model {
            total_bytes += std::fs::metadata(draft).map(|m| m.len()).unwrap_or(0);
        }

        total_bytes / (1024 * 1024)
    }

    /// Estimates KV cache + compute buffer VRAM overhead in MB based on `cfg.ctx_size`.
    pub fn estimate_kv_and_compute_mb(cfg: &ClusterConfig) -> u64 {
        ((cfg.ctx_size as u64) * 38) / 1024 + 900
    }

    /// Core N-Node Greedy Placement & Split Planner (pure function for easy unit testing).
    ///
    /// - Always checks if the model + KV cache fits in Queen's free VRAM (`<= 92%`) first; if so, recruits **0 workers**.
    /// - If user hardcoded `tensor_split` (e.g. `"35,65"`), recruits at most `split_parts - 1` workers so the Queen is never starved.
    /// - Otherwise, packs the Queen's local high-bandwidth GPU first, then greedily recruits
    ///   only the minimum subset of ranked workers ($1 \dots K-1$) required to hold the overflow,
    ///   reserving KV/compute buffer headroom on each worker so workers never OOM.
    /// - Emits `--tensor-split w_1,w_2,...,w_{k-1},queen` normalized to sum to 100.
    pub fn plan_cluster_placement_with_vram(
        cfg: &ClusterConfig,
        model_mb: u64,
        local_free_vram_mb: u64,
        ranked_workers: &[DiscoveredWorker],
    ) -> PlacementPlan {
        let kv_and_compute_mb = Self::estimate_kv_and_compute_mb(cfg);
        let total_required_mb = model_mb.saturating_add(kv_and_compute_mb);

        // 1. Check if 100% of model + KV cache fits inside Queen's local GPU VRAM (even if tensor_split was set)
        if cfg.auto_rpc
            && model_mb > 0
            && local_free_vram_mb > 0
            && total_required_mb <= (local_free_vram_mb * 92) / 100
        {
            info!(
                "N-Node Planner: model ({} MB) + KV/compute ({} MB) = {} MB fits in Queen VRAM ({} MB) -> recruiting 0 RPC workers.",
                model_mb, kv_and_compute_mb, total_required_mb, local_free_vram_mb
            );
            return PlacementPlan {
                selected_workers: Vec::new(),
                tensor_split: None,
            };
        }

        if ranked_workers.is_empty() {
            return PlacementPlan {
                selected_workers: Vec::new(),
                tensor_split: None,
            };
        }

        // If user explicitly hardcoded tensor_split in ilima.json, honor it safely by recruiting
        // at most (num_parts - 1) workers so the last split value always belongs to the Queen GPU.
        if let Some(manual_ts) = &cfg.tensor_split {
            let num_parts = manual_ts.split(',').filter(|s| !s.trim().is_empty()).count();
            let max_workers = num_parts.saturating_sub(1).max(1);
            let selected: Vec<String> = ranked_workers
                .iter()
                .take(max_workers)
                .map(|w| w.endpoint.clone())
                .collect();
            return PlacementPlan {
                selected_workers: selected,
                tensor_split: Some(manual_ts.clone()),
            };
        }

        // If we don't know model size or local VRAM, recruit top worker and let llama.cpp split
        if model_mb == 0 || local_free_vram_mb == 0 {
            return PlacementPlan {
                selected_workers: vec![ranked_workers[0].endpoint.clone()],
                tensor_split: None,
            };
        }

        // 2. Greedy Queen-First Packing:
        // Keep ~90% of Queen free VRAM usable, minus ~55% of the KV/compute overhead on Queen.
        // When built-in MTP (--spec-type draft-mtp) is enabled, reserve an extra 1950 MB on Queen
        // for the in-place blk.64 MTP draft context, h_nextn buffer, and multi-sequence recurrent state
        // so Vulkan0 never spills into WDDM shared system RAM (producing "32,68" on 16GB Queen + 8GB Worker).
        let mtp_queen_headroom_mb = if cfg.spec_type.as_deref() == Some("draft-mtp") {
            1950
        } else {
            0
        };
        let queen_usable_for_weights = (local_free_vram_mb * 90 / 100)
            .saturating_sub((kv_and_compute_mb * 55) / 100)
            .saturating_sub(mtp_queen_headroom_mb);
        let queen_weight_mb = queen_usable_for_weights
            .min(model_mb.saturating_sub(512))
            .max(model_mb / 4);

        let mut remaining_mb = model_mb.saturating_sub(queen_weight_mb);
        let mut selected_workers: Vec<String> = Vec::new();
        let mut worker_allocs_mb: Vec<u64> = Vec::new();

        for w in ranked_workers {
            if remaining_mb == 0 {
                break;
            }
            // If worker didn't report VRAM (e.g. standalone rpc-server on an 8GB laptop), assume ~6600 MB free
            let w_free = if w.available_vram_mb > 0 {
                w.available_vram_mb
            } else {
                6600
            };
            // Reserve ~550 MB on each worker for RPC compute graph buffers + KV cache share (prevents 8GB worker OOM)
            let w_usable_mb = ((w_free * 85) / 100).saturating_sub(550).max(512);
            let alloc_mb = remaining_mb.min(w_usable_mb);

            selected_workers.push(w.endpoint.clone());
            worker_allocs_mb.push(alloc_mb);
            remaining_mb = remaining_mb.saturating_sub(alloc_mb);
        }

        // 3. Build --tensor-split in llama-server order: [worker_1, ..., worker_{k-1}, queen]
        let mut raw_parts = worker_allocs_mb;
        raw_parts.push(queen_weight_mb);
        let total_alloc: u64 = raw_parts.iter().sum::<u64>().max(1);

        let mut pcts: Vec<u64> = raw_parts
            .iter()
            .map(|x| ((x * 100 + total_alloc / 2) / total_alloc).max(1))
            .collect();

        // Adjust the last element (Queen) so the percentages sum to exactly 100
        let sum_workers: u64 = pcts[..pcts.len() - 1].iter().sum();
        if sum_workers < 100 {
            if let Some(last) = pcts.last_mut() {
                *last = 100 - sum_workers;
            }
        }

        let split_str = pcts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");

        info!(
            "N-Node Planner: model={} MB, ctx={} -> recruited {}/{} workers {:?} with --tensor-split {}",
            model_mb,
            cfg.ctx_size,
            selected_workers.len(),
            ranked_workers.len(),
            selected_workers,
            split_str
        );

        PlacementPlan {
            selected_workers,
            tensor_split: Some(split_str),
        }
    }

    /// Wrapper that reads the `.gguf` file size on disk (including shards and draft model) and probes
    /// local VRAM before calling `plan_cluster_placement_with_vram`.
    /// Uses `total_vram_mb.max(free_vram_mb)` because Windows WDDM reserves ~885 MiB for the desktop compositor
    /// while `plan_cluster_placement_with_vram` already applies its own `90%` safety factor.
    pub fn plan_cluster_placement(
        cfg: &ClusterConfig,
        model_path: &str,
        ranked_workers: &[DiscoveredWorker],
    ) -> PlacementPlan {
        let model_mb = Self::estimate_total_model_mb(cfg, model_path);
        let (total_vram_mb, local_free_vram_mb, _) = Self::probe_local_vram_mb(cfg);
        let effective_queen_vram_mb = total_vram_mb.max(local_free_vram_mb);
        Self::plan_cluster_placement_with_vram(cfg, model_mb, effective_queen_vram_mb, ranked_workers)
    }

    /// Pre-flight check: verify every RPC worker port accepts a TCP connection within 2 seconds.
    pub async fn check_workers_reachable(workers: &[String]) -> Result<(), String> {
        for worker in workers {
            let connect_fut = tokio::net::TcpStream::connect(worker);
            match tokio::time::timeout(Duration::from_secs(2), connect_fut).await {
                Ok(Ok(_stream)) => {
                    info!("Pre-flight RPC check passed for worker {}", worker);
                }
                Ok(Err(e)) => {
                    return Err(format!(
                        "Worker {} is unreachable ({}). Is `ilima worker` running and port 50052 open in the firewall?",
                        worker, e
                    ));
                }
                Err(_) => {
                    return Err(format!(
                        "Timed out connecting to worker {} (2s). Check the worker IP and firewall.",
                        worker
                    ));
                }
            }
        }
        Ok(())
    }

    /// Wait until llama-server answers /health with 200, or until the timeout,
    /// or returns immediately false if the child process exited prematurely.
    pub async fn wait_until_healthy(port: u16, timeout_secs: u64) -> bool {
        Self::wait_until_healthy_with_child(port, timeout_secs, None).await
    }

    /// Wait until llama-server answers /health with 200, or aborts immediately if child exited.
    pub async fn wait_until_healthy_with_child(
        port: u16,
        timeout_secs: u64,
        child_lock: Option<std::sync::Arc<std::sync::Mutex<Option<Child>>>>,
    ) -> bool {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", port);
        let start = Instant::now();
        loop {
            // Check if child process has exited
            if let Some(ref lock) = child_lock {
                if let Ok(mut guard) = lock.lock() {
                    if let Some(child) = guard.as_mut() {
                        if let Ok(Some(status)) = child.try_wait() {
                            eprintln!("llama-server process exited prematurely: {}", status);
                            return false;
                        }
                    }
                }
            }

            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    return true;
                }
            }
            if start.elapsed() >= Duration::from_secs(timeout_secs) {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Sends a chat request to the running llama-server.
    /// Returns (answer text, tokens per second if llama-server reported it).
    pub async fn generate_completion(
        prompt: &str,
        server_port: u16,
        max_tokens: u32,
    ) -> Result<(String, Option<f64>), String> {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", server_port);

        let body = serde_json::json!({
            "messages": [
                { "role": "system", "content": "You are a helpful assistant. Keep any internal reasoning concise and directly provide the complete, detailed final answer." },
                { "role": "user", "content": prompt }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "cache_prompt": true
        });

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Failed to connect to llama-server: {}. Is it running?", e))?;

        if !resp.status().is_success() {
            let code = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("llama-server returned {}: {}", code, text));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let content = {
            let c = json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim();
            let r = json["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or("").trim();
            if !c.is_empty() && !r.is_empty() {
                format!("<think>\n{}\n</think>\n\n{}", r, c)
            } else if !c.is_empty() {
                c.to_string()
            } else if !r.is_empty() {
                r.to_string()
            } else {
                String::new()
            }
        };
        let tps = json["timings"]["predicted_per_second"].as_f64();
        Ok((content, tps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_workers() -> Vec<DiscoveredWorker> {
        vec![
            DiscoveredWorker {
                endpoint: "192.168.1.50:50052".to_string(),
                rtt_ms: 0.35,
                total_vram_mb: 8192,
                available_vram_mb: 7000,
                gpu_name: Some("NVIDIA GeForce RTX 5050 Laptop GPU".to_string()),
            },
            DiscoveredWorker {
                endpoint: "192.168.1.51:50052".to_string(),
                rtt_ms: 9.5,
                total_vram_mb: 8192,
                available_vram_mb: 7000,
                gpu_name: Some("Secondary Wi-Fi Node".to_string()),
            },
        ]
    }

    #[test]
    fn test_small_model_runs_100_percent_locally() {
        let mut cfg = ClusterConfig::default();
        cfg.ctx_size = 8192;
        cfg.tensor_split = Some("35,65".to_string()); // Even if manual split is set, small model runs locally!
        let workers = sample_workers();

        // 6.5 GB model on 16 GB Queen -> fits 100% locally, recruits 0 workers
        let plan = LlamaCppBackend::plan_cluster_placement_with_vram(&cfg, 6500, 16304, &workers);
        assert!(plan.selected_workers.is_empty());
        assert_eq!(plan.tensor_split, None);
    }

    #[test]
    fn test_manual_split_caps_recruited_workers_count() {
        let mut cfg = ClusterConfig::default();
        cfg.ctx_size = 64192;
        cfg.tensor_split = Some("35,65".to_string()); // 2-part split -> max 1 worker + 1 Queen
        let workers = sample_workers();

        let plan = LlamaCppBackend::plan_cluster_placement_with_vram(&cfg, 16500, 16304, &workers);
        assert_eq!(plan.selected_workers, vec!["192.168.1.50:50052".to_string()]);
        assert_eq!(plan.tensor_split, Some("35,65".to_string()));
    }

    #[test]
    fn test_27b_model_recruits_single_fastest_worker_with_22_78_split() {
        let mut cfg = ClusterConfig::default();
        cfg.ctx_size = 64192;
        let workers = sample_workers();

        // 16.5 GB model (16500 MB) + 64k ctx on 16304 MB Queen -> recruits ONLY first worker (22,78)
        let plan = LlamaCppBackend::plan_cluster_placement_with_vram(&cfg, 16500, 16304, &workers);
        assert_eq!(plan.selected_workers, vec!["192.168.1.50:50052".to_string()]);
        assert_eq!(plan.tensor_split, Some("22,78".to_string()));
    }

    #[test]
    fn test_27b_model_with_draft_mtp_computes_32_68_split() {
        let mut cfg = ClusterConfig::default();
        cfg.ctx_size = 64192;
        cfg.spec_type = Some("draft-mtp".to_string());
        let workers = sample_workers();

        // 15.66 GB model (16035 MB) + 64k ctx + built-in MTP on 16304 MB Queen -> 32,68 split
        let plan = LlamaCppBackend::plan_cluster_placement_with_vram(&cfg, 16035, 16304, &workers);
        assert_eq!(plan.selected_workers, vec!["192.168.1.50:50052".to_string()]);
        assert_eq!(plan.tensor_split, Some("32,68".to_string()));
    }

    #[test]
    fn test_70b_model_recruits_multiple_workers() {
        let mut cfg = ClusterConfig::default();
        cfg.ctx_size = 32768;
        let workers = vec![
            DiscoveredWorker {
                endpoint: "192.168.1.50:50052".to_string(),
                rtt_ms: 0.35,
                total_vram_mb: 12288,
                available_vram_mb: 11000,
                gpu_name: None,
            },
            DiscoveredWorker {
                endpoint: "192.168.1.51:50052".to_string(),
                rtt_ms: 0.45,
                total_vram_mb: 12288,
                available_vram_mb: 11000,
                gpu_name: None,
            },
            DiscoveredWorker {
                endpoint: "192.168.1.52:50052".to_string(),
                rtt_ms: 0.55,
                total_vram_mb: 8192,
                available_vram_mb: 7500,
                gpu_name: None,
            },
        ];

        // 36 GB model (36000 MB) on 16 GB Queen + 3 workers -> recruits all 3 workers
        let plan = LlamaCppBackend::plan_cluster_placement_with_vram(&cfg, 36000, 16304, &workers);
        assert_eq!(plan.selected_workers.len(), 3);
        assert!(plan.tensor_split.is_some());
        let parts: Vec<u64> = plan
            .tensor_split
            .unwrap()
            .split(',')
            .map(|s| s.parse::<u64>().unwrap())
            .collect();
        assert_eq!(parts.len(), 4); // 3 workers + 1 Queen
        assert_eq!(parts.iter().sum::<u64>(), 100);
    }

    #[test]
    fn test_parse_best_gpu_prefers_discrete_over_igpu() {
        let sample_output = "\
ggml_vulkan: Found 2 Vulkan devices:
Vulkan0: AMD Radeon(TM) Graphics (8192 MiB, 7900 MiB free)
Vulkan1: AMD Radeon RX 9070 XT (16384 MiB, 15800 MiB free)
";
        let (total, free, name) = LlamaCppBackend::parse_best_gpu_from_device_list(sample_output);
        assert_eq!(total, 16384);
        assert_eq!(free, 15800);
        assert_eq!(name.as_deref(), Some("AMD Radeon RX 9070 XT"));
    }

    #[test]
    fn test_parse_rpc_device_from_list() {
        let sample = "\
ggml_rpc: connecting to 192.168.1.50:50052
Available devices:
  RPC0[192.168.1.50:50052]: NVIDIA GeForce RTX 5050 Laptop GPU (8150 MiB, 7070 MiB free)
  Vulkan0: AMD Radeon RX 9070 XT (16384 MiB, 15800 MiB free)
";
        let parsed = LlamaCppBackend::parse_rpc_device_from_list(sample);
        assert_eq!(
            parsed,
            Some((
                8150,
                7070,
                Some("NVIDIA GeForce RTX 5050 Laptop GPU".to_string())
            ))
        );
    }

    #[test]
    fn test_is_physical_lan_ipv4_filters_virtual_subnets() {
        assert!(LlamaCppBackend::is_physical_lan_ipv4("192.168.1.10"));
        assert!(LlamaCppBackend::is_physical_lan_ipv4("10.0.0.15"));
        assert!(!LlamaCppBackend::is_physical_lan_ipv4("172.24.16.1")); // WSL2 / Hyper-V
        assert!(!LlamaCppBackend::is_physical_lan_ipv4("127.0.0.1"));
    }

    #[test]
    fn test_gpu_bandwidth_multiplier_penalizes_igpu() {
        assert!(LlamaCppBackend::gpu_bandwidth_multiplier(Some("AMD Radeon(TM) Graphics")) < 0.5);
        assert!(LlamaCppBackend::gpu_bandwidth_multiplier(Some("Intel(R) UHD Graphics 770")) < 0.5);
        assert!(LlamaCppBackend::gpu_bandwidth_multiplier(Some("AMD Radeon RX 9070 XT")) > 1.1);
        assert_eq!(LlamaCppBackend::gpu_bandwidth_multiplier(Some("NVIDIA GeForce RTX 5050 Laptop GPU")), 1.0);
    }
}