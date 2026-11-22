mod p2p;
mod storage;
mod compute;
mod scheduler;
mod inference;
mod http_api;
mod layer_cache;
mod inference_config;
mod cluster_config;
mod runtime_manager;

mod message;
mod model;
mod backend;

use clap::{Parser, Subcommand};
use libp2p::{
    core::upgrade,
    gossipsub, mdns, noise,
    swarm::SwarmEvent,
    tcp, yamux, PeerId, Transport,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;
use tracing::info;
use p2p::IlimaBehavior;
use storage::Storage;
use compute::ComputeEngine;
use scheduler::Scheduler;
use inference::InferenceEngine;
use futures::future::Either;
use futures::StreamExt;
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the coordinator node. Reads cluster settings from a JSON file.
    Start {
        #[arg(long, default_value = "ilima.json")]
        config: String,
    },
    /// Upload a file to the Ilima cluster
    Upload {
        path: String,
    },
    /// Retrieve a file from the Ilima cluster
    Get {
        cid: String,
    },
    /// Run a compute task (Matrix Multiplication)
    Compute {
        size: usize,
    },
    /// Run inference on a model
    Infer {
        #[arg(long)]
        model: String, // Path to model file
        #[arg(long)]
        tokenizer: String, // Path to tokenizer file
        #[arg(long)]
        prompt: String,
    },
    /// Setup the agent environment (builds llama.cpp in WSL)
    Setup,
    /// Start as a Worker (lends this machine's GPU via rpc-server)
    Worker {
        #[arg(long, default_value_t = 50052)]
        port: u16,
        #[arg(long, default_value = "ilima.json")]
        config: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let storage = Storage::new(".ilima/storage").await?;
    let scheduler = Arc::new(Mutex::new(Scheduler::new()));

    match args.command {
        Some(Commands::Upload { path }) => {
            let data = tokio::fs::read(&path).await?;
            let cid = storage.store(&data).await?;
            println!("Uploaded file. CID: {}", cid);
            return Ok(());
        }
        Some(Commands::Get { cid }) => {
            if let Some(data) = storage.retrieve(&cid).await? {
                let filename = format!("download_{}", &cid[0..8]);
                tokio::fs::write(&filename, data).await?;
                println!("Retrieved file to {}", filename);
            } else {
                println!("File not found locally.");
            }
            return Ok(());
        }
        Some(Commands::Compute { size }) => {
            println!("Generating {}x{} matrices...", size, size);
            let matrix_a = ComputeEngine::generate_matrix(size, size);
            let matrix_b = ComputeEngine::generate_matrix(size, size);

            println!("Serializing and storing matrices...");
            let data_a = ComputeEngine::serialize_matrix(&matrix_a)?;
            let data_b = ComputeEngine::serialize_matrix(&matrix_b)?;
            
            let cid_a = storage.store(&data_a).await?;
            let cid_b = storage.store(&data_b).await?;
            
            println!("Stored Matrix A: {}", cid_a);
            println!("Stored Matrix B: {}", cid_b);

            println!("Computing locally for verification...");
            let start = std::time::Instant::now();
            let result = ComputeEngine::multiply(&matrix_a, &matrix_b)?;
            let duration = start.elapsed();
            
            println!("Computation complete in {:.2?}", duration);
            let data_res = ComputeEngine::serialize_matrix(&result)?;
            let cid_res = storage.store(&data_res).await?;
            println!("Result stored at: {}", cid_res);
            
            return Ok(());
        }
        Some(Commands::Infer { model, tokenizer, prompt }) => {
            println!("Loading model from {}...", model);
            let mut engine = InferenceEngine::load(&model, &tokenizer, None)?;
            println!("Generating...");
            let output = engine.generate(&prompt, 50)?;
            println!("Output: {}{}", prompt, output);
            return Ok(());
        }
        Some(Commands::Setup) => {
            backend::llama_cpp::LlamaCppBackend::setup().map_err(|e| e.to_string())?;
            return Ok(());
        }
        Some(Commands::Worker { .. }) | None | Some(Commands::Start { .. }) => {
            // Continue to unified cluster config & P2P swarm startup below
        }
    }

    info!("Starting Ilima...");

    // Initialize Inference Engine (shared state)
    let inference_engine = Arc::new(Mutex::new(None));

    // Channel for internal messages (e.g. inference results to broadcast)
    let (tx, mut rx) = tokio::sync::mpsc::channel::<message::Message>(32);

    // Shared state for pending requests (for Coordinator to wait for results)
    let pending_requests = Arc::new(Mutex::new(std::collections::HashMap::<String, tokio::sync::oneshot::Sender<Result<String, String>>>::new()));

    // Start HTTP API in a separate task
    let api_engine = inference_engine.clone();
    let api_scheduler = scheduler.clone();
    let api_tx = tx.clone();
    let api_pending = pending_requests.clone();
    
    // Load cluster settings (ilima.json by default)
    let (config_path, is_worker, worker_cli_port) = match &args.command {
        Some(Commands::Start { config }) => (config.clone(), false, None),
        Some(Commands::Worker { config, port }) => (config.clone(), true, Some(*port)),
        _ => ("ilima.json".to_string(), false, None),
    };
    let mut cluster_config = cluster_config::ClusterConfig::load(&config_path);
    if let Some(p) = worker_cli_port {
        cluster_config.rpc_port = p;
    }
    runtime_manager::ensure_runtime_ready(&mut cluster_config).await;
    if is_worker {
        // Dedicated workers never start llama-server; they run rpc-server in a self-healing supervisor loop.
        cluster_config.model_path = None;
        let worker_cfg = cluster_config.clone();
        let rpc_port = worker_cfg.rpc_port;
        std::thread::spawn(move || {
            loop {
                if !runtime_manager::is_local_port_open(rpc_port) {
                    info!("Starting Worker rpc-server on port {}...", rpc_port);
                    if let Err(e) = backend::llama_cpp::LlamaCppBackend::start_worker(&worker_cfg, rpc_port) {
                        eprintln!("Worker rpc-server exited ({}); restarting in 2s...", e);
                    }
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        });
    } else if cluster_config.auto_start_rpc_worker
        && cluster_config.model_path.is_none()
        && !runtime_manager::is_local_port_open(cluster_config.rpc_port)
    {
        // Symmetric node startup: also run rpc-server in the background if port 50052 is free
        let bg_cfg = cluster_config.clone();
        let rpc_port = bg_cfg.rpc_port;
        std::thread::spawn(move || {
            info!("Auto-starting background rpc-server on port {}...", rpc_port);
            let _ = backend::llama_cpp::LlamaCppBackend::start_worker(&bg_cfg, rpc_port);
        });
    }

    let hb_config = cluster_config.clone();
    let hb_tx = tx.clone();
    tokio::spawn(async move {
        let (total_vram_mb, available_vram_mb, gpu_name) =
            backend::llama_cpp::LlamaCppBackend::probe_local_vram_mb(&hb_config);
        let mut seq: u64 = 0;
        loop {
            seq = seq.wrapping_add(1);
            let lan_ips = runtime_manager::detect_local_ipv4s();
            let caps = scheduler::NodeCapabilities {
                total_ram_mb: 16384,
                available_ram_mb: 8192,
                total_vram_mb,
                available_vram_mb,
                cpu_cores: num_cpus::get(),
                has_gpu: total_vram_mb > 0,
                gpu_name: gpu_name.clone(),
                lan_ips,
                rpc_port: hb_config.rpc_port,
                heartbeat_seq: seq,
            };
            let msg = message::Message::NodeHeartbeat {
                rpc_port: hb_config.rpc_port,
                capabilities: caps,
            };
            let _ = hb_tx.send(msg).await;
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    });

    let shutdown_cfg = cluster_config.clone();

    if !is_worker {
        tokio::spawn(async move {
            http_api::start_server(api_engine, api_scheduler, api_tx, api_pending, cluster_config).await;
        });
    }

    // Create a random PeerId
    let id_keys = libp2p::identity::Keypair::generate_ed25519();
    let peer_id = PeerId::from(id_keys.public());
    info!("Local peer id: {peer_id}");

    // Set up the transport
    let tcp_transport = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true))
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::Config::new(&id_keys).unwrap())
        .multiplex(yamux::Config::default())
        .boxed();
    
    let ws_transport = libp2p::websocket::WsConfig::new(tcp::tokio::Transport::new(tcp::Config::default().nodelay(true)))
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::Config::new(&id_keys).unwrap())
        .multiplex(yamux::Config::default())
        .boxed();

    let transport = tcp_transport.or_transport(ws_transport)
        .map(|either, _| match either {
            Either::Left((peer_id, muxer)) => (peer_id, libp2p::core::muxing::StreamMuxerBox::new(muxer)),
            Either::Right((peer_id, muxer)) => (peer_id, libp2p::core::muxing::StreamMuxerBox::new(muxer)),
        })
        .boxed();

    // Set up the behaviour
    let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;
    
    // Gossipsub configuration
    let message_id_fn = |message: &gossipsub::Message| {
        let mut s = DefaultHasher::new();
        message.data.hash(&mut s);
        gossipsub::MessageId::from(s.finish().to_string())
    };
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1)) // Faster heartbeat for testing
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(message_id_fn)
        .mesh_n_low(0)
        .mesh_n(2)
        .mesh_n_high(4)
        .mesh_outbound_min(0) 
        .flood_publish(true) // Ensure it pushes even if mesh is empty
        .build()
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    let gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(id_keys),
        gossipsub_config,
    )?;

    let behaviour = IlimaBehavior {
        gossipsub,
        mdns,
    };

    // Build the Swarm
    let mut swarm = libp2p::Swarm::new(transport, behaviour, peer_id, libp2p::swarm::Config::with_tokio_executor());

    // Listen on all interfaces
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    // Subscribe to gossipsub topic
    let topic = gossipsub::IdentTopic::new("ilima-main");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    // Event loop
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl+C received — stopping child llama-server processes and exiting gracefully...");
                if !is_worker {
                    backend::llama_cpp::LlamaCppBackend::kill_stale_servers(&shutdown_cfg);
                }
                break;
            }
            internal_msg = rx.recv() => {
                if let Some(msg) = internal_msg {
                    if let Ok(data) = serde_json::to_vec(&msg) {
                        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(topic.clone(), data) {
                             info!("Failed to publish message: {:?}", e);
                        }
                    }
                }
            }
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        info!("Listening on {address:?}");
                    }

                    SwarmEvent::Behaviour(p2p::IlimaBehaviorEvent::Mdns(mdns::Event::Discovered(list))) => {
                        for (peer_id, multiaddr) in list {
                            info!("mDNS discovered a new peer: {peer_id} at {multiaddr}");
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                            let _ = swarm.dial(multiaddr.clone());
                            scheduler.lock().unwrap().add_peer(peer_id, multiaddr);
                        }
                    }
                    SwarmEvent::Behaviour(p2p::IlimaBehaviorEvent::Mdns(mdns::Event::Expired(list))) => {
                        for (peer_id, _multiaddr) in list {
                            info!("mDNS discover peer has expired: {peer_id}");
                            swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                            scheduler.lock().unwrap().remove_peer(&peer_id);
                        }
                    }
                    SwarmEvent::Behaviour(p2p::IlimaBehaviorEvent::Gossipsub(gossipsub::Event::Message {
                        propagation_source: peer_id,
                        message_id: _id,
                        message,
                    })) => {
                        // Deserialize message
                        if let Ok(msg) = serde_json::from_slice::<message::Message>(&message.data) {
                            info!("Received P2P message from {}: {:?}", peer_id, msg);
                            
                            match msg {
                                message::Message::TaskRequest { task_id, prompt, model_name, download_url, layer_range } => {
                                    info!("Processing Task {} (Range: {:?})...", task_id, layer_range);
                                    let engine = inference_engine.clone();
                                    let tx_inner = tx.clone();
                                    
                                    tokio::spawn(async move {
                                         let model_path = format!("models/{}", model_name);
                                         
                                         // LAZY LOADING: Check if model exists, if not, try download
                                         if !std::path::Path::new(&model_path).exists() {
                                             if let Some(url) = download_url {
                                                 info!("Model missing. Attempting to download from Coordinator: {}", url);
                                                 match reqwest::get(&url).await {
                                                     Ok(resp) => {
                                                         if resp.status().is_success() {
                                                             // Stream download
                                                             use futures::StreamExt;
                                                             if let Ok(file) = std::fs::File::create(&model_path) {
                                                                 let mut file = std::io::BufWriter::new(file);
                                                                 let mut stream = resp.bytes_stream();
                                                                 while let Some(item) = stream.next().await {
                                                                     if let Ok(chunk) = item {
                                                                         let _ = std::io::Write::write_all(&mut file, &chunk);
                                                                     }
                                                                 }
                                                                 // Flush
                                                                 let _ = std::io::Write::flush(&mut file);
                                                                 info!("Download complete: {}", model_path);
                                                             }
                                                         } else {
                                                             info!("Coordinator failed to serve model (Status {})", resp.status());
                                                         }
                                                     }
                                                     Err(e) => info!("Download error: {}", e),
                                                 }
                                             }
                                         }

                                         let res = tokio::task::spawn_blocking(move || {
                                             let mut lock = engine.lock().unwrap();
                                             // Check if loaded, if not try to load
                                            if lock.is_none() || lock.as_ref().unwrap().model_path != model_path {
                                                // Check for specific tokenizer
                                                let specific_tok = format!("{}.tokenizer.json", model_path);
                                                let tokenizer_path = if std::path::Path::new(&specific_tok).exists() {
                                                    specific_tok
                                                } else {
                                                    "tokenizer.json".to_string()
                                                };

                                                 if std::path::Path::new(&model_path).exists() {
                                                     info!("Loading model {} with range {:?}...", model_name, layer_range);
                                                     if let Ok(new_engine) = InferenceEngine::load(&model_path, &tokenizer_path, layer_range) {
                                                         *lock = Some(new_engine);
                                                     }
                                                 }
                                            }
                                            
                                            if let Some(eng) = lock.as_mut() {
                                                eng.generate(&prompt, 50).map_err(|e| e.to_string())
                                            } else {
                                                Err("Model not found or failed to load (Download might have failed)".to_string())
                                            }
                                         }).await;
                                         
                                         match res {
                                             Ok(Ok(output)) => {
                                                 let response = message::Message::TaskResponse {
                                                     task_id,
                                                     result: Ok(output),
                                                 };
                                                 let _ = tx_inner.send(response).await;
                                             },
                                             Ok(Err(e)) => {
                                                  let response = message::Message::TaskResponse {
                                                     task_id,
                                                     result: Err(e),
                                                 };
                                                 let _ = tx_inner.send(response).await;
                                             }
                                             _ => {}
                                         }
                                    });
                                }
                                message::Message::TaskResponse { task_id, result } => {
                                    info!("Result received for Task {}", task_id);
                                    let mut pending = pending_requests.lock().unwrap();
                                    if let Some(sender) = pending.remove(&task_id) {
                                        let _ = sender.send(result);
                                    }
                                }
                                message::Message::NodeHeartbeat { rpc_port, mut capabilities } => {
                                    let origin_peer = message.source.unwrap_or(peer_id);
                                    if capabilities.rpc_port == 0 {
                                        capabilities.rpc_port = rpc_port;
                                    }
                                    info!(
                                        "Peer {} heartbeat (seq={}): GPU={:?}, free VRAM={} MB, IPs={:?}",
                                        origin_peer,
                                        capabilities.heartbeat_seq,
                                        capabilities.gpu_name,
                                        capabilities.available_vram_mb,
                                        capabilities.lan_ips
                                    );
                                    scheduler.lock().unwrap().update_peer_capabilities(&origin_peer, capabilities);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

