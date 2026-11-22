use libp2p::{PeerId, Multiaddr};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeCapabilities {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub total_vram_mb: u64,
    pub available_vram_mb: u64,
    pub cpu_cores: usize,
    pub has_gpu: bool,
    pub gpu_name: Option<String>,
    /// All active non-loopback IPv4 addresses on this node (e.g. Ethernet + Wi-Fi).
    pub lan_ips: Vec<String>,
    /// The port where this node's `rpc-server` is listening (default 50052).
    pub rpc_port: u16,
    /// Monotonically increasing sequence number so Gossipsub never drops periodic heartbeats as duplicates.
    pub heartbeat_seq: u64,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            total_ram_mb: 8192,
            available_ram_mb: 4096,
            total_vram_mb: 0,
            available_vram_mb: 0,
            cpu_cores: num_cpus::get(),
            has_gpu: false,
            gpu_name: None,
            lan_ips: Vec::new(),
            rpc_port: 50052,
            heartbeat_seq: 0,
        }
    }
}

/// A reachable RPC worker node evaluated by the Queen's RTT & VRAM ranker.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredWorker {
    /// Fastest reachable `ip:port` endpoint for this worker (e.g. "192.168.1.50:50052").
    pub endpoint: String,
    /// Measured TCP connect round-trip time in milliseconds (lower = faster link, e.g. Ethernet < 1ms).
    pub rtt_ms: f64,
    /// Total GPU VRAM in MB (0 if unknown / standalone rpc-server).
    pub total_vram_mb: u64,
    /// Available GPU VRAM in MB (0 if unknown / standalone rpc-server).
    pub available_vram_mb: u64,
    /// Optional GPU name reported via heartbeat.
    pub gpu_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: PeerId,
    pub address: Vec<Multiaddr>,
    pub status: String, // "active", "busy", "offline"
    pub capabilities: NodeCapabilities,
    pub last_seen: std::time::Instant,
    pub current_load: f32, // 0.0 to 1.0
}

pub struct Scheduler {
    pub peers: HashMap<PeerId, PeerInfo>,
}

#[allow(dead_code)]
impl Scheduler {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }

    /// Add or update a peer with optional capabilities
    pub fn add_peer(&mut self, peer_id: PeerId, addr: Multiaddr) {
        self.add_peer_with_capabilities(peer_id, addr, None);
    }

    /// Add or update a peer with capabilities
    pub fn add_peer_with_capabilities(
        &mut self, 
        peer_id: PeerId, 
        addr: Multiaddr,
        capabilities: Option<NodeCapabilities>
    ) {
        // Clone for use in closure
        let caps_clone = capabilities.clone();
        
        let entry = self.peers.entry(peer_id).or_insert_with(|| PeerInfo {
            id: peer_id,
            address: Vec::new(),
            status: "active".to_string(),
            capabilities: caps_clone.unwrap_or_default(),
            last_seen: std::time::Instant::now(),
            current_load: 0.0,
        });
        
        if !entry.address.contains(&addr) {
            entry.address.push(addr);
        }
        
        // Update capabilities if provided
        if let Some(caps) = capabilities {
            entry.capabilities = caps;
        }
        
        entry.last_seen = std::time::Instant::now();
    }

    /// Update peer capabilities
    pub fn update_peer_capabilities(&mut self, peer_id: &PeerId, capabilities: NodeCapabilities) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.capabilities = capabilities;
            peer.last_seen = std::time::Instant::now();
        }
    }

    /// Update peer load
    pub fn update_peer_load(&mut self, peer_id: &PeerId, load: f32) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.current_load = load.clamp(0.0, 1.0);
            peer.last_seen = std::time::Instant::now();
        }
    }

    pub fn remove_peer(&mut self, peer_id: &PeerId) {
        self.peers.remove(peer_id);
    }

    /// Get the best available peer based on capabilities and current load
    pub fn get_best_peer(&self) -> Option<PeerId> {
        self.peers
            .values()
            .filter(|p| p.status == "active")
            .min_by(|a, b| {
                // Score based on: current load (primary) and available memory (secondary)
                let score_a = a.current_load + (1.0 - (a.capabilities.available_ram_mb as f32 / 
                    a.capabilities.total_ram_mb.max(1) as f32)) * 0.3;
                let score_b = b.current_load + (1.0 - (b.capabilities.available_ram_mb as f32 / 
                    b.capabilities.total_ram_mb.max(1) as f32)) * 0.3;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id)
    }

    /// Get the best peer for GPU-intensive tasks
    pub fn get_best_gpu_peer(&self) -> Option<PeerId> {
        self.peers
            .values()
            .filter(|p| p.status == "active" && p.capabilities.has_gpu && p.capabilities.available_vram_mb > 1024)
            .min_by_key(|p| (p.current_load * 100.0) as u32)
            .map(|p| p.id)
    }

    /// Get any available peer (fallback to simple selection)
    pub fn get_available_peer(&self) -> Option<PeerId> {
        self.get_best_peer().or_else(|| {
            self.peers
                .values()
                .find(|p| p.status == "active")
                .map(|p| p.id)
        })
    }

    /// Get peer statistics
    pub fn get_stats(&self) -> SchedulerStats {
        let total_peers = self.peers.len();
        let active_peers = self.peers.values().filter(|p| p.status == "active").count();
        let gpu_peers = self.peers.values().filter(|p| p.capabilities.has_gpu).count();
        
        let avg_load = if active_peers > 0 {
            self.peers.values()
                .filter(|p| p.status == "active")
                .map(|p| p.current_load)
                .sum::<f32>() / active_peers as f32
        } else {
            0.0
        };

        SchedulerStats {
            total_peers,
            active_peers,
            gpu_peers,
            avg_load,
        }
    }

    /// Remove stale peers (not seen in last 5 minutes)
    pub fn cleanup_stale_peers(&mut self) {
        let stale_timeout = std::time::Duration::from_secs(300); // 5 minutes
        let now = std::time::Instant::now();
        
        self.peers.retain(|_, peer| {
            now.duration_since(peer.last_seen) < stale_timeout
        });
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_peers: usize,
    pub active_peers: usize,
    pub gpu_peers: usize,
    pub avg_load: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::PeerId;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = Scheduler::new();
        assert_eq!(scheduler.peers.len(), 0);
    }

    #[test]
    fn test_add_peer() {
        let mut scheduler = Scheduler::new();
        let peer_id = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/4001".parse().unwrap();
        
        scheduler.add_peer(peer_id, addr);
        assert_eq!(scheduler.peers.len(), 1);
    }

    #[test]
    fn test_best_peer_selection() {
        let mut scheduler = Scheduler::new();
        
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/4001".parse().unwrap();
        
        scheduler.add_peer(peer1, addr.clone());
        scheduler.add_peer(peer2, addr);
        
        // Set different loads
        scheduler.update_peer_load(&peer1, 0.8);
        scheduler.update_peer_load(&peer2, 0.3);
        
        // peer2 should be selected (lower load)
        let best = scheduler.get_best_peer().unwrap();
        assert_eq!(best, peer2);
    }
}
