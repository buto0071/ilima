use axum::{
    extract::{State, Json, Multipart, DefaultBodyLimit},
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;
use crate::inference::InferenceEngine;
use crate::scheduler::Scheduler;
use crate::message::Message;
use crate::backend::llama_cpp::LlamaCppBackend;
use crate::inference_config::InferenceConfig;
use crate::cluster_config::ClusterConfig;
use std::io::Write;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};


#[derive(Clone)]
pub struct AppState {
    pub inference_engine: Arc<Mutex<Option<InferenceEngine>>>,
    pub scheduler: Arc<Mutex<Scheduler>>,
    #[allow(dead_code)]
    pub p2p_sender: mpsc::Sender<Message>,
    #[allow(dead_code)]
    pub pending_requests: Arc<Mutex<HashMap<String, oneshot::Sender<Result<String, String>>>>>,
    pub server_process: Arc<Mutex<Option<std::process::Child>>>,
    pub current_config: Arc<Mutex<Option<ServerConfig>>>,
    pub model_status: Arc<Mutex<ModelStatus>>,
    pub optimization_config: Arc<Mutex<InferenceConfig>>,
    pub cluster: Arc<Mutex<ClusterConfig>>,
    /// Only one question is sent to llama-server at a time; others wait here (a simple queue).
    pub inference_queue: Arc<tokio::sync::Mutex<()>>,
    /// Serializes model loading so concurrent triggers (startup, watcher, API, OpenAI auto-switch) never race.
    pub load_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone, PartialEq, serde::Serialize)]
pub enum ModelStatus {
    Offline,
    Loading,
    Active,
    Error(String),
}

/// What is currently loaded in llama-server.
#[derive(Clone, PartialEq)]
pub struct ServerConfig {
    pub model_path: String,
    pub workers: Vec<String>,
    pub worker_details: Vec<crate::scheduler::DiscoveredWorker>,
    pub tensor_split: Option<String>,
}

/// Stop any running llama-server, autonomously discover & rank N-node workers,
/// compute the greedy placement plan for `model_path`, and wait until llama-server is healthy.
async fn load_model_and_wait(state: AppState, model_path: String) {
    // 0. Serialize all load attempts so concurrent callers never kill each other's llama-server
    let _load_guard = state.load_lock.lock().await;

    let mut cfg = state.cluster.lock().unwrap().clone();
    crate::runtime_manager::ensure_runtime_ready(&mut cfg).await;
    state.cluster.lock().unwrap().llama_bin_dir = cfg.llama_bin_dir.clone();

    let peers_empty = state.scheduler.lock().unwrap().peers.is_empty();
    if cfg.workers.is_empty() && peers_empty {
        println!("No static workers in ilima.json; waiting up to 2s for mDNS peer discovery...");
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    *state.model_status.lock().unwrap() = ModelStatus::Loading;

    // 1. Stop the old server FIRST so its VRAM is freed before we probe local free VRAM!
    {
        let mut guard = state.server_process.lock().unwrap();
        if let Some(mut child) = guard.take() {
            println!("Stopping current llama-server...");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    LlamaCppBackend::kill_stale_servers(&cfg);

    // Periodically prune stale Gossipsub peers while refreshing cluster placement
    state.scheduler.lock().unwrap().cleanup_stale_peers();

    // Take a Send-safe snapshot of peers before awaiting TCP RTT probes
    let peers_snapshot: Vec<crate::scheduler::PeerInfo> = {
        let sched = state.scheduler.lock().unwrap();
        sched.peers.values().cloned().collect()
    };
    let local_ips = crate::runtime_manager::detect_local_ipv4s();

    // 2. Discover & rank all reachable workers across all interfaces (Ethernet > Wi-Fi)
    let ranked_workers =
        LlamaCppBackend::discover_and_rank_workers(&cfg, &peers_snapshot, &local_ips).await;

    // 3. Compute N-node greedy placement plan (recruits 0..K-1 workers + normalized tensor_split)
    let plan = LlamaCppBackend::plan_cluster_placement(&cfg, &model_path, &ranked_workers);
    let workers = plan.selected_workers;
    let worker_details: Vec<crate::scheduler::DiscoveredWorker> = ranked_workers
        .into_iter()
        .filter(|w| workers.contains(&w.endpoint))
        .collect();
    if workers.is_empty() {
        cfg.tensor_split = None;
    } else if cfg.tensor_split.is_none() {
        cfg.tensor_split = plan.tensor_split;
    }

    // 4. Pre-flight check: verify every selected RPC worker is reachable
    if !workers.is_empty() {
        if let Err(err_msg) = LlamaCppBackend::check_workers_reachable(&workers).await {
            println!("Pre-flight worker check failed: {}", err_msg);
            *state.model_status.lock().unwrap() = ModelStatus::Error(err_msg);
            return;
        }
    }

    *state.current_config.lock().unwrap() = Some(ServerConfig {
        model_path: model_path.clone(),
        workers: workers.clone(),
        worker_details,
        tensor_split: cfg.tensor_split.clone(),
    });

    println!(
        "Starting llama-server for {} with workers {:?} and tensor_split {:?}",
        model_path, workers, cfg.tensor_split
    );
    match LlamaCppBackend::start_server(&cfg, &model_path, &workers) {
        Ok(child) => {
            *state.server_process.lock().unwrap() = Some(child);
            println!("Waiting up to {} seconds for the model to load...", cfg.load_timeout_secs);
            let ok = LlamaCppBackend::wait_until_healthy_with_child(
                cfg.server_port,
                cfg.load_timeout_secs,
                Some(state.server_process.clone()),
            ).await;
            let mut status = state.model_status.lock().unwrap();
            if ok {
                println!("Model {} loaded and ready.", model_path);
                *status = ModelStatus::Active;
            } else {
                *status = ModelStatus::Error(
                    "llama-server exited or failed to become healthy. Check console logs for details."
                        .to_string(),
                );
            }
        }
        Err(e) => {
            println!("Failed to start llama-server: {}", e);
            *state.model_status.lock().unwrap() = ModelStatus::Error(format!("Failed to start: {}", e));
        }
    }
}

pub async fn start_server(
    inference_engine: Arc<Mutex<Option<InferenceEngine>>>,
    scheduler: Arc<Mutex<Scheduler>>,
    p2p_sender: mpsc::Sender<Message>,
    pending_requests: Arc<Mutex<HashMap<String, oneshot::Sender<Result<String, String>>>>>,
    cluster: ClusterConfig,
) {
    let state = AppState {
        inference_engine,
        scheduler,
        p2p_sender,
        pending_requests,
        server_process: Arc::new(Mutex::new(None)),
        current_config: Arc::new(Mutex::new(None)),
        model_status: Arc::new(Mutex::new(ModelStatus::Offline)),
        optimization_config: Arc::new(Mutex::new(InferenceConfig::default())),
        cluster: Arc::new(Mutex::new(cluster.clone())),
        inference_queue: Arc::new(tokio::sync::Mutex::new(())),
        load_lock: Arc::new(tokio::sync::Mutex::new(())),
    };

    // Auto-load the model from the config file (loaded ONCE, kept in memory)
    if let Some(model_path) = cluster.model_path.clone() {
        println!("Auto-loading model from config: {}", model_path);
        let load_state = state.clone();
        tokio::spawn(async move {
            load_model_and_wait(load_state, model_path).await;
        });
    }

    // Background cluster supervisor & hot-recruit watcher:
    // 1. Monitors the active llama-server child process; if a worker disconnects and llama-server
    //    exits unexpectedly, automatically recovers by reloading on the remaining healthy nodes.
    // 2. If the Queen loaded an overflowing model locally (0 workers) because the Worker laptop
    //    wasn't turned on yet, and a Worker joins the LAN later, automatically reloads once to recruit it.
    let watcher_state = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(12)).await;

            // Check 1: Did an Active llama-server crash or exit because an RPC worker dropped?
            let crashed_model = {
                let status = watcher_state.model_status.lock().unwrap().clone();
                let current = watcher_state.current_config.lock().unwrap().clone();
                let mut proc_guard = watcher_state.server_process.lock().unwrap();
                match (status, current, proc_guard.as_mut()) {
                    (ModelStatus::Active, Some(c), Some(child)) => {
                        if let Ok(Some(exit_status)) = child.try_wait() {
                            eprintln!(
                                "Supervisor: Active llama-server exited unexpectedly ({}). Auto-recovering {}...",
                                exit_status, c.model_path
                            );
                            Some(c.model_path)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            };

            if let Some(recover_path) = crashed_model {
                load_model_and_wait(watcher_state.clone(), recover_path).await;
                continue;
            }

            // Check 2: Hot-recruit only if active with 0 workers, inference_queue is idle,
            //          AND the model actually overflows local VRAM
            if watcher_state.inference_queue.try_lock().is_err() {
                // An inference request or SSE stream is currently in flight -> never interrupt it!
                continue;
            }

            let (is_active_with_zero_workers, active_model_path, cfg) = {
                let status = watcher_state.model_status.lock().unwrap().clone();
                let current = watcher_state.current_config.lock().unwrap().clone();
                let cfg = watcher_state.cluster.lock().unwrap().clone();
                match (status, current) {
                    (ModelStatus::Active, Some(c)) if c.workers.is_empty() => {
                        (true, c.model_path, cfg)
                    }
                    _ => (false, String::new(), cfg),
                }
            };

            if is_active_with_zero_workers {
                let model_mb = LlamaCppBackend::estimate_total_model_mb(&cfg, &active_model_path);
                let kv_and_compute_mb = LlamaCppBackend::estimate_kv_and_compute_mb(&cfg);
                // Use total_vram_mb (not free_vram_mb while loaded!) to check if model already fits locally
                let (total_vram_mb, _, _) = LlamaCppBackend::probe_local_vram_mb(&cfg);
                if total_vram_mb > 0
                    && model_mb > 0
                    && model_mb + kv_and_compute_mb <= (total_vram_mb * 92) / 100
                {
                    // Model fits 100% in local VRAM -> do not sweep or hot-recruit
                    continue;
                }

                let peers_snapshot: Vec<crate::scheduler::PeerInfo> = {
                    let sched = watcher_state.scheduler.lock().unwrap();
                    sched.peers.values().cloned().collect()
                };
                let local_ips = crate::runtime_manager::detect_local_ipv4s();
                let ranked =
                    LlamaCppBackend::discover_and_rank_workers(&cfg, &peers_snapshot, &local_ips)
                        .await;
                if !ranked.is_empty() {
                    let plan = LlamaCppBackend::plan_cluster_placement_with_vram(
                        &cfg,
                        model_mb,
                        total_vram_mb,
                        &ranked,
                    );
                    if !plan.selected_workers.is_empty() {
                        println!(
                            "Hot-Recruit: New GPU worker(s) {:?} detected on LAN for overflowing model! Upgrading cluster to --tensor-split {:?}...",
                            plan.selected_workers, plan.tensor_split
                        );
                        load_model_and_wait(watcher_state.clone(), active_model_path).await;
                    }
                }
            }
        }
    });

    // Create models directory if it doesn't exist
    let _ = std::fs::create_dir_all("models");

    let dashboard_dir = ["ilima-dashboard/out", "ilima-dashboard/dist", "dashboard"]
        .into_iter()
        .find(|d| std::path::Path::new(d).exists())
        .unwrap_or("models");

    let app = Router::new()
        .route("/api/status", get(get_status))
        .route("/api/models", get(list_models))
        .route("/api/models/load", post(load_model_handler))
        .route("/api/peers", get(list_peers))
        .route("/api/inference", post(run_inference))
        .route("/v1/models", get(openai_list_models))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/api/upload", post(upload_model))
        .route("/api/cluster", get(get_cluster))
        .route("/api/cluster/reload", post(reload_cluster_handler))
        .route("/api/optimization/config", get(get_optimization_config))
        .route("/api/optimization/config", post(set_optimization_config))
        .route("/api/optimization/stats", get(get_optimization_stats))
        .route("/api/scheduler/stats", get(get_scheduler_stats))
        .route("/download/{filename}", get(download_agent_binary))
        .route("/join", get(join_worker_script))
        .nest_service("/runtime", tower_http::services::ServeDir::new(cluster.llama_bin_dir.clone()))
        .nest_service("/models", tower_http::services::ServeDir::new("models"))
        .fallback_service(tower_http::services::ServeDir::new(dashboard_dir))
        .layer(DefaultBodyLimit::disable())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Dashboard API listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn get_cluster(State(state): State<AppState>) -> Json<Value> {
    let cfg = state.cluster.lock().unwrap().clone();
    let loaded = state.current_config.lock().unwrap().clone();
    Json(json!({
        "config": serde_json::to_value(&cfg).unwrap_or(json!({})),
        "loaded_model": loaded.as_ref().map(|c| c.model_path.clone()),
        "active_workers": loaded.as_ref().map(|c| c.workers.clone()).unwrap_or_default(),
        "active_tensor_split": loaded.and_then(|c| c.tensor_split),
    }))
}

async fn reload_cluster_handler(State(state): State<AppState>) -> Json<Value> {
    if matches!(*state.model_status.lock().unwrap(), ModelStatus::Loading)
        || state.load_lock.try_lock().is_err()
    {
        println!("Cluster reload requested, but a model load is already in progress. Skipping duplicate reload.");
        return Json(json!({ "status": "already_loading" }));
    }
    let model_to_reload = {
        let current = state.current_config.lock().unwrap().clone();
        let cfg = state.cluster.lock().unwrap().clone();
        current
            .map(|c| c.model_path)
            .or(cfg.model_path)
    };
    if let Some(model_path) = model_to_reload {
        println!("Cluster reload requested (worker joined/updated): reloading {}...", model_path);
        *state.model_status.lock().unwrap() = ModelStatus::Loading;
        let load_state = state.clone();
        tokio::spawn(async move {
            load_model_and_wait(load_state, model_path).await;
        });
        Json(json!({ "status": "reloading" }))
    } else {
        Json(json!({ "status": "no_model_selected" }))
    }
}

async fn get_status(State(state): State<AppState>) -> Json<Value> {
    let peers = state.scheduler.lock().unwrap().peers.len();
    let current_config = state.current_config.lock().unwrap().clone();
    let active_model = current_config.as_ref().map(|c| c.model_path.clone());
    let active_workers = current_config.as_ref().map(|c| c.workers.clone()).unwrap_or_default();
    let active_tensor_split = current_config.and_then(|c| c.tensor_split);
    let model_status = state.model_status.lock().unwrap().clone();
    
    // Normalize path to just filename for UI
    let active_model_name = active_model.map(|path| {
        std::path::Path::new(&path)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    });

    Json(json!({
        "node_id": "local-node",
        "role": "Coordinator",
        "peers": peers,
        "status": "active",
        "active_model": active_model_name,
        "active_workers": active_workers,
        "active_tensor_split": active_tensor_split,
        "model_status": model_status
    }))
}

async fn list_peers(State(state): State<AppState>) -> Json<Value> {
    let scheduler = state.scheduler.lock().unwrap();
    let current_cfg = state.current_config.lock().unwrap().clone();
    let mut peers: Vec<Value> = scheduler.peers.values().map(|p| {
        json!({
            "id": p.id.to_string(),
            "address": p.address.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", "),
            "lan_ips": p.capabilities.lan_ips,
            "rpc_port": p.capabilities.rpc_port,
            "gpu_name": p.capabilities.gpu_name,
            "available_vram_mb": p.capabilities.available_vram_mb,
            "total_vram_mb": p.capabilities.total_vram_mb,
            "role": "Worker",
            "latency": 1,
            "status": p.status
        })
    }).collect();

    // Also include any active standalone RPC workers discovered via the /24 port 50052 sweep
    if let Some(cfg) = current_cfg {
        for worker_endpoint in cfg.workers {
            let ip_only = worker_endpoint.split(':').next().unwrap_or("");
            let port_num: u16 = worker_endpoint
                .rsplit_once(':')
                .and_then(|(_, p)| p.parse::<u16>().ok())
                .unwrap_or(50052);
            let detail = cfg
                .worker_details
                .iter()
                .find(|d| d.endpoint == worker_endpoint);
            let already_listed = scheduler.peers.values().any(|p| {
                p.capabilities.lan_ips.iter().any(|ip| ip == ip_only)
                    || p.address.iter().any(|a| a.to_string().contains(ip_only))
            });
            if !already_listed {
                peers.push(json!({
                    "id": format!("rpc-{}", worker_endpoint),
                    "address": worker_endpoint,
                    "lan_ips": [ip_only],
                    "rpc_port": port_num,
                    "gpu_name": detail.and_then(|d| d.gpu_name.clone()).unwrap_or_else(|| "Active RPC Worker".to_string()),
                    "available_vram_mb": detail.map(|d| d.available_vram_mb).unwrap_or(0),
                    "total_vram_mb": detail.map(|d| d.total_vram_mb).unwrap_or(0),
                    "role": "Worker",
                    "latency": detail.map(|d| d.rtt_ms).unwrap_or(1.0),
                    "status": "active"
                }));
            }
        }
    }

    Json(json!({ 
        "peers": peers,
        "metrics": {
            "cpu_usage": 0.0,
            "total_mem": 0,
            "used_mem": 0,
            "gpu_usage": 0
        }
    }))
}

async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let cfg = state.cluster.lock().unwrap().clone();
    let models: Vec<String> = cfg
        .discover_models()
        .into_iter()
        .map(|(fname, _)| fname)
        .collect();
    Json(json!({ "models": models }))
}

async fn upload_model(mut multipart: Multipart) -> Json<Value> {
    let mut repo_id: Option<String> = None;
    let mut file_name: Option<String> = None;
    
    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some(name) = field.name() {
            match name {
                "repo_id" => {
                    if let Ok(val) = field.text().await {
                        if !val.is_empty() {
                            println!("Received HF Repo ID: {}", val);
                            repo_id = Some(val);
                        }
                    }
                }
                "model" => {
                    if let Some(raw_fname) = field.file_name() {
                        // Sanitize filename to prevent path traversal (`../../`)
                        let safe_fname = std::path::Path::new(raw_fname)
                            .file_name()
                            .map(|f| f.to_string_lossy().to_string())
                            .unwrap_or_default();
                        if safe_fname.is_empty() {
                            return Json(json!({ "error": "Invalid model filename" }));
                        }
                        let path = format!("models/{}", safe_fname);
                        file_name = Some(safe_fname);
                        
                        // Read bytes
                         let data = match field.bytes().await {
                            Ok(bytes) => bytes,
                            Err(e) => return Json(json!({ "error": format!("Failed to read bytes: {}", e) })),
                        };

                        if let Ok(mut file) = std::fs::File::create(&path) {
                            if file.write_all(&data).is_err() {
                                 return Json(json!({ "error": "Failed to write model file" }));
                            }
                            println!("Uploaded model: {}", path);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // After processing fields, if we have both model name and repo_id, fetch tokenizer
    if let (Some(model_name), Some(repo)) = (file_name.clone(), repo_id) {
        println!("Attempting to auto-download tokenizer from {}", repo);
        tokio::spawn(async move {
            let url = format!("https://huggingface.co/{}/resolve/main/tokenizer.json", repo);
            let target_path = format!("models/{}.tokenizer.json", model_name);
            let client = reqwest::Client::new();
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    if let Ok(bytes) = resp.bytes().await {
                         let _ = std::fs::write(target_path, bytes);
                    }
                }
            }
        });
    }

    if let Some(fname) = file_name {
        Json(json!({ "status": "uploaded", "filename": fname }))
    } else {
        Json(json!({ "error": "No model file found in request" }))
    }
}

#[derive(serde::Deserialize)]
struct LoadModelRequest {
    model_path: String,
}

/// True if both paths point to the same model file (compares full path or file name).
fn same_model(a: &str, b: &str) -> bool {
    let name_a = std::path::Path::new(a).file_name().map(|f| f.to_string_lossy().to_string());
    let name_b = std::path::Path::new(b).file_name().map(|f| f.to_string_lossy().to_string());
    a == b || (name_a.is_some() && name_a == name_b)
}

async fn load_model_handler(
    State(state): State<AppState>,
    Json(payload): Json<LoadModelRequest>,
) -> Json<Value> {
    let cfg = state.cluster.lock().unwrap().clone();
    let model_path = cfg.resolve_model_path(&payload.model_path);
    println!("Request to load model: {}", model_path);

    // Do NOT reload if this model is already loaded or loading.
    let already_loaded = {
        let current = state.current_config.lock().unwrap();
        let status = state.model_status.lock().unwrap();
        match (&*current, &*status) {
            (Some(c), ModelStatus::Active) => same_model(&c.model_path, &model_path),
            (Some(c), ModelStatus::Loading) => same_model(&c.model_path, &model_path),
            _ => false,
        }
    };
    if already_loaded {
        println!("Model already loaded/loading. Skipping reload.");
        return Json(json!({ "status": "already_loaded", "message": "Model is already loaded (or loading). Not reloading." }));
    }

    *state.model_status.lock().unwrap() = ModelStatus::Loading;
    let load_state = state.clone();
    tokio::spawn(async move {
        load_model_and_wait(load_state, model_path).await;
    });

    Json(json!({ "status": "loading", "message": "Model load initiated" }))
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct InferenceRequest {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    prompt: String,
}

async fn run_inference(
    State(state): State<AppState>,
    Json(payload): Json<InferenceRequest>,
) -> Json<Value> {
    println!("Received inference request: {}", payload.prompt);

    let status_now = state.model_status.lock().unwrap().clone();
    match status_now {
        ModelStatus::Active => {}
        ModelStatus::Loading => {
            return Json(json!({ "error": "Model is still loading. Please wait." }));
        }
        ModelStatus::Offline => {
            return Json(json!({ "error": "No model loaded. Please select a model." }));
        }
        ModelStatus::Error(e) => {
            // The model may have finished loading after the timeout. Check once.
            let port = state.cluster.lock().unwrap().server_port;
            if LlamaCppBackend::wait_until_healthy(port, 1).await {
                println!("llama-server is healthy now. Recovering to Active.");
                *state.model_status.lock().unwrap() = ModelStatus::Active;
            } else {
                return Json(json!({ "error": format!("Model is not ready: {}", e) }));
            }
        }
    }

    let (port, max_tokens) = {
        let c = state.cluster.lock().unwrap();
        (c.server_port, c.max_tokens)
    };

    // Simple queue: wait for our turn instead of rejecting the request.
    let _turn = state.inference_queue.lock().await;

    let started = std::time::Instant::now();
    match LlamaCppBackend::generate_completion(&payload.prompt, port, max_tokens).await {
        Ok((text, tps)) => {
            let seconds = started.elapsed().as_secs_f64();
            println!("Generated {} chars in {:.1}s ({:?} tok/s)", text.len(), seconds, tps);
            Json(json!({
                "content": text,
                "result": text,
                "tokens_per_second": tps,
                "seconds": seconds
            }))
        }
        Err(e) => {
            println!("Inference error: {}", e);
            Json(json!({ "error": e }))
        }
    }
}

async fn openai_list_models(State(state): State<AppState>) -> Json<Value> {
    let cfg = state.cluster.lock().unwrap().clone();
    let discovered = cfg.discover_models();
    let mut data: Vec<Value> = discovered
        .into_iter()
        .map(|(fname, _full)| {
            json!({
                "id": fname,
                "object": "model",
                "owned_by": "ilima-mesh"
            })
        })
        .collect();

    if data.is_empty() {
        let loaded = state
            .current_config
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.model_path.clone())
            .unwrap_or_else(|| "ilima-default".to_string());
        data.push(json!({
            "id": loaded,
            "object": "model",
            "owned_by": "ilima-mesh"
        }));
    }

    Json(json!({
        "object": "list",
        "data": data
    }))
}

async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let is_stream = payload.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // Hold an owned mutex guard so streaming responses keep the queue locked until the stream finishes!
    let turn = state.inference_queue.clone().lock_owned().await;

    // Optional automatic model switching if the request specifies a different model that exists on disk
    if let Some(req_model) = payload.get("model").and_then(|v| v.as_str()) {
        if !req_model.is_empty() && req_model != "ilima-default" {
            let resolved = {
                let cfg = state.cluster.lock().unwrap();
                cfg.resolve_model_path(req_model)
            };
            if std::path::Path::new(&resolved).exists() {
                let (needs_switch, is_loading_same) = {
                    let current = state.current_config.lock().unwrap();
                    let status = state.model_status.lock().unwrap();
                    match (&*current, &*status) {
                        (Some(c), ModelStatus::Active) => (!same_model(&c.model_path, &resolved), false),
                        (Some(c), ModelStatus::Loading) if same_model(&c.model_path, &resolved) => (false, true),
                        _ => (true, false),
                    }
                };
                if is_loading_same {
                    // Wait for the in-progress load of this exact model to finish instead of restarting it
                    let _wait_load = state.load_lock.lock().await;
                } else if needs_switch {
                    println!("OpenAI API requested model '{}'; switching to {}...", req_model, resolved);
                    load_model_and_wait(state.clone(), resolved).await;
                }
            }
        }
    }

    // If a startup or hot-recruit model load is still in progress (e.g. when model="ilima-default"), wait for it
    if *state.model_status.lock().unwrap() == ModelStatus::Loading {
        let _wait_load = state.load_lock.lock().await;
    }

    let port = state.cluster.lock().unwrap().server_port;
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let client = reqwest::Client::new();
    let upstream = match client.post(&url).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({ "error": { "message": format!("llama-server unreachable: {}", e) } })),
            )
                .into_response();
        }
    };

    if is_stream {
        use futures::StreamExt;
        // Move `turn` into the stream closure so `inference_queue` stays locked until SSE completes
        let stream = upstream.bytes_stream().map(move |chunk| {
            let _keep_alive = &turn;
            chunk
        });
        axum::response::Response::builder()
            .status(axum::http::StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(axum::body::Body::from_stream(stream))
            .unwrap()
    } else {
        let status = axum::http::StatusCode::from_u16(upstream.status().as_u16())
            .unwrap_or(axum::http::StatusCode::OK);
        let body: Value = upstream.json().await.unwrap_or_else(|_| json!({}));
        drop(turn);
        (status, Json(body)).into_response()
    }
}

// Optimization configuration endpoints

#[derive(serde::Deserialize)]
struct SetOptimizationConfigRequest {
    preset: Option<String>, // "high_performance", "low_memory", "balanced"
    config: Option<InferenceConfig>,
}

async fn get_optimization_config(State(state): State<AppState>) -> Json<Value> {
    let config = state.optimization_config.lock().unwrap().clone();
    
    Json(json!({
        "config": {
            "quantization": {
                "bits": format!("{:?}", config.quantization.bits),
                "enabled": config.quantization.enabled,
                "block_size": config.quantization.block_size,
                "memory_reduction": config.quantization.memory_reduction_factor(),
                "expected_speedup": config.quantization.expected_speedup(),
            },
            "enable_prefetch": config.enable_prefetch,
            "cache_size_mb": config.cache_size_mb,
            "max_batch_size": config.max_batch_size,
        }
    }))
}

async fn set_optimization_config(
    State(state): State<AppState>,
    Json(payload): Json<SetOptimizationConfigRequest>,
) -> Json<Value> {
    let new_config = if let Some(preset) = payload.preset {
        match preset.as_str() {
            "high_performance" => InferenceConfig::high_performance(),
            "low_memory" => InferenceConfig::low_memory(),
            "balanced" => InferenceConfig::balanced(),
            _ => return Json(json!({ "error": "Invalid preset. Use: high_performance, low_memory, or balanced" })),
        }
    } else if let Some(config) = payload.config {
        config
    } else {
        return Json(json!({ "error": "Must provide either 'preset' or 'config'" }));
    };

    // Validate config
    if let Err(e) = new_config.validate() {
        return Json(json!({ "error": format!("Invalid configuration: {}", e) }));
    }

    // Update stored config
    {
        let mut config = state.optimization_config.lock().unwrap();
        *config = new_config.clone();
    }

    println!("Updated optimization configuration:");
    println!("{}", new_config.summary());

    Json(json!({
        "status": "updated",
        "config": {
            "quantization": {
                "bits": format!("{:?}", new_config.quantization.bits),
                "enabled": new_config.quantization.enabled,
                "memory_reduction": new_config.quantization.memory_reduction_factor(),
                "expected_speedup": new_config.quantization.expected_speedup(),
            },
            "enable_prefetch": new_config.enable_prefetch,
            "cache_size_mb": new_config.cache_size_mb,
            "max_batch_size": new_config.max_batch_size,
        }
    }))
}

async fn get_optimization_stats(State(state): State<AppState>) -> Json<Value> {
    let engine_guard = state.inference_engine.lock().unwrap();
    
    let cache_stats = if let Some(engine) = engine_guard.as_ref() {
        engine.cache_stats().map(|stats| {
            json!({
                "entries": stats.entries,
                "size_mb": stats.size_bytes as f64 / 1024.0 / 1024.0,
                "max_size_mb": stats.max_size_bytes as f64 / 1024.0 / 1024.0,
                "utilization": (stats.size_bytes as f64 / stats.max_size_bytes.max(1) as f64) * 100.0,
            })
        })
    } else {
        None
    };

    Json(json!({
        "cache": cache_stats.unwrap_or(json!({ "status": "no_engine_loaded" })),
    }))
}

async fn get_scheduler_stats(State(state): State<AppState>) -> Json<Value> {
    let scheduler = state.scheduler.lock().unwrap();
    let stats = scheduler.get_stats();

    let peers_detail: Vec<Value> = scheduler.peers.values().map(|p| {
        json!({
            "id": p.id.to_string(),
            "status": p.status,
            "load": p.current_load,
            "capabilities": {
                "total_ram_mb": p.capabilities.total_ram_mb,
                "available_ram_mb": p.capabilities.available_ram_mb,
                "total_vram_mb": p.capabilities.total_vram_mb,
                "available_vram_mb": p.capabilities.available_vram_mb,
                "cpu_cores": p.capabilities.cpu_cores,
                "has_gpu": p.capabilities.has_gpu,
                "gpu_name": p.capabilities.gpu_name,
            }
        })
    }).collect();

    Json(json!({
        "total_peers": stats.total_peers,
        "active_peers": stats.active_peers,
        "gpu_peers": stats.gpu_peers,
        "avg_load": stats.avg_load,
        "peers": peers_detail,
    }))
}

/// Streams the running `ilima.exe` binary and its companion MinGW runtime DLLs over the LAN.
async fn download_agent_binary(
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let allowed = [
        "ilima.exe",
        "libstdc++-6.dll",
        "libgcc_s_seh-1.dll",
        "libwinpthread-1.dll",
    ];
    if !allowed.contains(&filename.as_str()) {
        return (
            axum::http::StatusCode::NOT_FOUND,
            format!("File not allowed: {}", filename),
        )
            .into_response();
    }

    let exe_path = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to locate running executable: {}", e),
            )
                .into_response();
        }
    };

    let file_path = if filename == "ilima.exe" {
        exe_path
    } else {
        exe_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(&filename)
    };

    match tokio::fs::read(&file_path).await {
        Ok(bytes) => axum::response::Response::builder()
            .status(axum::http::StatusCode::OK)
            .header("content-type", "application/octet-stream")
            .header(
                "content-disposition",
                format!("attachment; filename=\"{}\"", filename),
            )
            .body(axum::body::Body::from(bytes))
            .unwrap(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read {}: {}", filename, e),
        )
            .into_response(),
    }
}

/// Returns a PowerShell one-liner bootstrap script (`irm http://<coordinator-ip>:3000/join | iex`)
/// that downloads `ilima.exe` and its runtime DLLs, detects any existing `rpc-server.exe`
/// on the worker PC, and starts the worker node.
async fn join_worker_script(headers: axum::http::HeaderMap) -> axum::response::Response {
    // Prefer the exact IP/host the worker used in `irm http://<coordinator-ip>:3000/join`
    let host_ip = headers
        .get(axum::http::header::HOST)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.split(':').next())
        .filter(|ip| !ip.is_empty() && *ip != "localhost" && *ip != "127.0.0.1")
        .map(|s| s.to_string());

    let coordinator_ip = host_ip.unwrap_or_else(|| {
        let local_ips = crate::runtime_manager::detect_local_ipv4s();
        local_ips
            .into_iter()
            .find(|ip| crate::backend::llama_cpp::LlamaCppBackend::is_physical_lan_ipv4(ip))
            .unwrap_or_else(|| "127.0.0.1".to_string())
    });

    let script = format!(
        r#"$ErrorActionPreference = 'Stop'
$WorkDir = Join-Path $HOME "ilima-worker"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Set-Location $WorkDir
Get-Process ilima, rpc-server, ggml-rpc-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300
Write-Host "[Ilima] Downloading ilima.exe and runtime DLLs from Coordinator ({coordinator_ip}:3000)..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "http://{coordinator_ip}:3000/download/ilima.exe" -OutFile "ilima.exe"
Unblock-File -Path "ilima.exe" -ErrorAction SilentlyContinue
foreach ($f in @("libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll")) {{
    try {{
        Invoke-WebRequest -Uri "http://{coordinator_ip}:3000/download/$f" -OutFile $f -ErrorAction Stop
        Unblock-File -Path $f -ErrorAction SilentlyContinue
    }} catch {{}}
}}
New-NetFirewallRule -DisplayName "Ilima Worker" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 50052,4001-4100 -Profile Private,Domain -ErrorAction SilentlyContinue | Out-Null
New-NetFirewallRule -DisplayName "Ilima mDNS" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 5353 -Profile Private,Domain -ErrorAction SilentlyContinue | Out-Null

$RuntimeDir = Join-Path $WorkDir "runtime"
$LocalCudaDir = Get-ChildItem -Path "$HOME\Downloads", "C:\Users\*\Downloads" -Directory -Filter "llama-b12503*cuda*" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if ($LocalCudaDir -and (Test-Path "$LocalCudaDir\ggml-rpc-server.exe")) {{
    Write-Host "[Ilima] Detected native CUDA b12503 runtime at $LocalCudaDir" -ForegroundColor Green
    $SelectedBinDir = $LocalCudaDir
}} else {{
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    Write-Host "[Ilima] Syncing matching Vulkan GPU runtime (b12503) from Coordinator ({coordinator_ip}:3000/runtime)..." -ForegroundColor Cyan
    $RuntimeFiles = @(
        "ggml-rpc-server.exe", "ggml-rpc.dll", "ggml.dll", "ggml-base.dll", "ggml-vulkan.dll",
        "ggml-cpu-x64.dll", "ggml-cpu-zen4.dll", "ggml-cpu-haswell.dll", "libomp.dll",
        "llama-server.exe", "llama-server-impl.dll", "llama-common.dll", "llama.dll", "mtmd.dll"
    )
    foreach ($rf in $RuntimeFiles) {{
        $dest = Join-Path $RuntimeDir $rf
        if (-not (Test-Path $dest)) {{
            Invoke-WebRequest -Uri "http://{coordinator_ip}:3000/runtime/$rf" -OutFile $dest -ErrorAction SilentlyContinue
            Unblock-File -Path $dest -ErrorAction SilentlyContinue
        }}
    }}
    $SelectedBinDir = $RuntimeDir
}}
$DevOut = (& "$SelectedBinDir\llama-server.exe" --list-devices 2>&1) -join "`n"
Write-Host "[Ilima] Verified GPU runtime ready:" -ForegroundColor Green
Write-Host $DevOut -ForegroundColor Gray

$Json = @{{ llama_bin_dir = $SelectedBinDir; use_wsl = $false; rpc_port = 50052 }} | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $WorkDir "ilima.json"), $Json)
Write-Host "[Ilima] Starting Ilima Worker Node on port 50052..." -ForegroundColor Green
Start-Job -ScriptBlock {{
    Start-Sleep -Seconds 4
    try {{ Invoke-RestMethod -Method Post -Uri "http://{coordinator_ip}:3000/api/cluster/reload" | Out-Null }} catch {{}}
}} | Out-Null
& .\ilima.exe worker
if ($LASTEXITCODE -ne 0) {{
    Write-Host "[Ilima] Worker exited with code: $LASTEXITCODE" -ForegroundColor Red
}}
"#
    );

    axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header("content-type", "text/plain; charset=utf-8")
        .body(axum::body::Body::from(script))
        .unwrap()
}
