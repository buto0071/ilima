use serde::{Deserialize, Serialize};
use std::path::Path;

/// Settings for running one big model across several machines with llama.cpp.
/// Loaded from a JSON file (default: `ilima.json` in the working directory).
/// Any field missing from the JSON file uses the default value below.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusterConfig {
    /// Folder that contains llama-server(.exe) and rpc-server(.exe).
    /// Native Windows example: "C:\\llama.cpp"
    /// WSL example (must be an absolute Linux path, no $HOME): "/home/me/llama.cpp/build/bin"
    pub llama_bin_dir: String,
    /// true = run the llama.cpp binaries inside WSL. false = run native binaries.
    pub use_wsl: bool,
    /// Model to load automatically when `ilima start` runs. null = wait for the dashboard.
    pub model_path: Option<String>,
    /// Directories scanned for .gguf models (used by /api/models and /v1/models).
    pub models_dirs: Vec<String>,
    /// Worker rpc-server addresses, e.g. ["192.168.1.50:50052"].
    /// If empty, Ilima uses peers discovered on the network with `rpc_port`.
    pub workers: Vec<String>,
    /// Port used by rpc-server on workers.
    pub rpc_port: u16,
    /// Port used by llama-server on the Queen.
    pub server_port: u16,
    /// Value for --tensor-split, e.g. "22,78". If null, Ilima computes an N-node split automatically.
    pub tensor_split: Option<String>,
    /// Number of layers to put on GPUs (99 = all).
    pub ngl: u32,
    /// Context size in tokens.
    pub ctx_size: u32,
    /// Optional small draft model for speculative decoding (same model family as the main model, or standalone MTP GGUF).
    pub draft_model: Option<String>,
    pub draft_max: u32,
    pub draft_min: u32,
    /// Device for the draft model, e.g. "Vulkan0". null = auto-pin to Queen's best local discrete GPU.
    pub draft_device: Option<String>,
    /// Speculative decoding type for llama-server (--spec-type), e.g. "draft-mtp", "draft-simple", "ngram-mod".
    pub spec_type: Option<String>,
    /// Reasoning effort passed to llama-server chat template (--reasoning-effort): "minimal", "low", "medium", "high", "xhigh".
    pub reasoning_effort: Option<String>,
    /// Token budget for thinking (--reasoning-budget): -1 for unrestricted, 0 to disable, N > 0 for cap.
    pub reasoning_budget: Option<i32>,
    /// For MoE models: keep the experts of this many layers on the CPU (--n-cpu-moe).
    pub n_cpu_moe: Option<u32>,
    /// Number of parallel request slots in llama-server.
    pub parallel_slots: u32,
    /// Value for --cache-reuse (reuse chat history between questions). 0 = do not pass the flag.
    pub cache_reuse: u32,
    /// How long to wait for llama-server to finish loading (seconds).
    pub load_timeout_secs: u64,
    /// Max tokens generated per answer.
    pub max_tokens: u32,
    /// Extra raw arguments appended to the llama-server command line.
    pub extra_server_args: Vec<String>,
    /// Queen's local GPU VRAM in MB (0 = auto-detect via `llama-server --list-devices`).
    pub local_vram_mb: u64,
    /// If true, only use --rpc workers when the model is too large to fit in local_vram_mb.
    pub auto_rpc: bool,
    /// If true, automatically start a local rpc-server in the background when ilima starts.
    pub auto_start_rpc_worker: bool,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            llama_bin_dir: "C:\\llama.cpp".to_string(),
            use_wsl: false,
            model_path: None,
            models_dirs: vec!["models".to_string()],
            workers: Vec::new(),
            rpc_port: 50052,
            server_port: 8081,
            tensor_split: None,
            ngl: 99,
            ctx_size: 8192,
            draft_model: None,
            draft_max: 3,
            draft_min: 0,
            draft_device: None,
            spec_type: None,
            reasoning_effort: None,
            reasoning_budget: None,
            n_cpu_moe: None,
            parallel_slots: 1,
            cache_reuse: 256,
            load_timeout_secs: 900,
            max_tokens: 1024,
            extra_server_args: Vec::new(),
            local_vram_mb: 0,
            auto_rpc: true,
            auto_start_rpc_worker: true,
        }
    }
}

impl ClusterConfig {
    /// Load the config from a JSON file. If the file is missing or invalid, use defaults.
    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let clean = text.trim_start_matches('\u{feff}');
                match serde_json::from_str::<ClusterConfig>(clean) {
                    Ok(cfg) => {
                        println!("Loaded cluster config from {}", path);
                        cfg
                    }
                    Err(e) => {
                        eprintln!("Invalid cluster config {}: {}. Using defaults.", path, e);
                        Self::default()
                    }
                }
            }
            Err(_) => {
                println!("No cluster config found at {}. Using defaults.", path);
                Self::default()
            }
        }
    }

    /// Full path to a llama.cpp binary, e.g. binary("llama-server").
    pub fn binary(&self, name: &str) -> String {
        if self.use_wsl {
            format!("{}/{}", self.llama_bin_dir.trim_end_matches('/'), name)
        } else {
            let file_name = if cfg!(windows) {
                format!("{}.exe", name)
            } else {
                name.to_string()
            };
            let primary = Path::new(&self.llama_bin_dir).join(&file_name);
            if !primary.exists() && name == "rpc-server" {
                let alt = Path::new(&self.llama_bin_dir).join(if cfg!(windows) {
                    "ggml-rpc-server.exe"
                } else {
                    "ggml-rpc-server"
                });
                if alt.exists() {
                    return alt.to_string_lossy().to_string();
                }
            }
            primary.to_string_lossy().to_string()
        }
    }

    /// Discover all `.gguf` and `.bin` models across `models_dirs` and `model_path`.
    /// Returns a deduplicated list of `(file_name, full_path_string)`.
    pub fn discover_models(&self) -> Vec<(String, String)> {
        let mut found: Vec<(String, String)> = Vec::new();
        let mut dirs = self.models_dirs.clone();
        if !dirs.iter().any(|d| d == "models") {
            dirs.push("models".to_string());
        }
        if let Some(mp) = &self.model_path {
            let p = Path::new(mp);
            if p.exists() {
                if let Some(fname) = p.file_name().map(|f| f.to_string_lossy().to_string()) {
                    found.push((fname, mp.clone()));
                }
                if let Some(parent) = p.parent() {
                    let parent_str = parent.to_string_lossy().to_string();
                    if !parent_str.is_empty() && !dirs.contains(&parent_str) {
                        dirs.push(parent_str);
                    }
                }
            }
        }

        for dir in dirs {
            Self::scan_dir_for_models(Path::new(&dir), true, &mut found);
        }
        found
    }

    fn scan_dir_for_models(dir: &Path, recurse_one_level: bool, found: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(fname) = path.file_name().and_then(|f| f.to_str()) {
                    if fname.ends_with(".gguf") || fname.ends_with(".bin") {
                        // Skip multimodal vision projector files (mmproj-*.gguf), standalone MTP draft heads (mtp-*.gguf), and non-initial shards (-00002-of-00003.gguf)
                        let lower_fname = fname.to_ascii_lowercase();
                        if lower_fname.starts_with("mmproj") || lower_fname.starts_with("mtp-") {
                            continue;
                        }
                        if fname.contains("-of-") && !fname.contains("-00001-of-") {
                            continue;
                        }
                        let full = path.to_string_lossy().to_string();
                        if !found.iter().any(|(existing_name, _)| existing_name == fname) {
                            found.push((fname.to_string(), full));
                        }
                    }
                }
            } else if recurse_one_level && path.is_dir() {
                Self::scan_dir_for_models(&path, false, found);
            }
        }
    }

    /// Resolve a model name or path (e.g. from `/v1/chat/completions` or `/api/models/load`)
    /// to an existing file path on disk.
    pub fn resolve_model_path(&self, raw: &str) -> String {
        if Path::new(raw).exists() {
            return raw.to_string();
        }
        for (fname, full_path) in self.discover_models() {
            if fname.eq_ignore_ascii_case(raw) || full_path.eq_ignore_ascii_case(raw) {
                return full_path;
            }
        }
        format!("models/{}", raw)
    }
}