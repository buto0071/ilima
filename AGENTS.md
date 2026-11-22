# AGENTS.md — Automated Setup & Verification Guide for Ilima

Use this playbook when asked to build, configure, start, or verify Ilima on a user's Coordinator and Worker machine(s).

## Architecture & Ports
- **Coordinator (`ilima start --config ilima.json`):**
  - `TCP 3000` — HTTP API, OpenAI `/v1/*` proxy, Web Dashboard, and Windows worker `/join` bootstrap.
  - `TCP 8081` — Managed `llama-server` instance (configurable via `server_port`).
  - `TCP 4001-4100`, `UDP 5353` — libp2p / mDNS peer discovery.
- **Worker (`ilima worker --config ilima.json`):**
  - `TCP 50052` — `ggml-rpc-server` / `rpc-server` (configurable via `rpc_port`).
- **Version invariant:** Coordinator and every Worker MUST run the exact same `llama.cpp` tag (`b12503`). `ilima` auto-downloads `b12503` Vulkan (Windows/Linux) or Metal (macOS) into `./runtime` if `llama_bin_dir` is missing.

## Step 1: Check Prerequisites
Verify tools before building:
```powershell
rustc --version
cargo --version
node --version   # optional, only needed for ilima-dashboard
```

## Step 2: Build
From the repo root:
```powershell
cargo build --release -p ilima
```
Optional dashboard build (only if Node.js is installed; `ilima` serves `ilima-dashboard/dist` automatically):
```powershell
Push-Location ilima-dashboard
npm ci
npm run build
Pop-Location
```

## Step 3: Create `ilima.json` on the Coordinator
1. Copy `ilima.example.json` to `ilima.json` (do not overwrite an existing `ilima.json` without asking the user).
2. Ensure a `.gguf` model is available:
   - Place `.gguf` files in `./models/`, OR add the user's model folder(s) to `models_dirs` in `ilima.json`.
   - Optionally set `model_path` to auto-load a model on startup (leaving `model_path: null` starts idle until a model is loaded via `/api/models/load`, `/v1/chat/completions`, or the dashboard).
3. Leave `"llama_bin_dir": "./runtime"` so `b12503` auto-downloads on first run, unless the user already has a `llama.cpp` `b12503` directory (then point `llama_bin_dir` there).
4. Leave `"workers": []`, `"tensor_split": null`, and `"local_vram_mb": 0` for automatic LAN discovery, VRAM probing, and split calculation.
5. **Only if** the user's model has a built-in MTP head (such as `Qwen3.8-27B-Q4_K_M.gguf`), enable MTP in `ilima.json`:
   ```json
   "ctx_size": 64192,
   "spec_type": "draft-mtp",
   "draft_max": 2,
   "draft_min": 0,
   "reasoning_effort": "medium",
   "parallel_slots": 1,
   "extra_server_args": ["-fa", "on"]
   ```
   Do **not** enable `draft-mtp` on models that do not ship embedded `nextn` MTP tensors.

## Step 4: Start the Coordinator & Open Firewall (Windows)
Ensure Windows Firewall allows inbound `TCP 3000` on `Private,Domain` if workers will use `/join`:
```powershell
.\target\release\ilima.exe start --config ilima.json
```
Wait for `Dashboard API listening on http://0.0.0.0:3000` in stdout. Note: first launch may take a minute while `./runtime` downloads `llama.cpp` `b12503` from GitHub.

## Step 5: Connect a Worker (Optional, for Multi-GPU Splitting)
- **Windows worker (recommended):** Ask the user (or run on the worker machine) in PowerShell, replacing `<COORDINATOR_LAN_IP>` with the Coordinator's physical LAN IP (`192.168.x.x` or `10.x.x.x`, NOT Tailscale `100.x` or WSL `172.x`):
  ```powershell
  irm http://<COORDINATOR_LAN_IP>:3000/join | iex
  ```
  - For NVIDIA workers, optionally extract the official `llama-b12503-bin-win-cuda-*` zip into the worker's `Downloads` folder first so `/join` selects CUDA automatically instead of Vulkan.
- **Manual worker:** Copy `ilima.worker.example.json` to `ilima.json` on the worker and run:
  ```powershell
  .\target\release\ilima.exe worker --config ilima.json
  ```

## Step 6: Verify Everything Works
Run these checks against the Coordinator:
```powershell
# 1. Confirm API is up and check active_workers / active_tensor_split / model_status
Invoke-RestMethod http://localhost:3000/api/status

# 2. Confirm discovered .gguf models
Invoke-RestMethod http://localhost:3000/v1/models

# 3. Confirm detected LAN workers (if using a second machine)
Invoke-RestMethod http://localhost:3000/api/peers

# 4. Smoke-test inference (replace <MODEL_FILENAME> with an id from /v1/models)
$body = @{
  model = "<MODEL_FILENAME>"
  messages = @(@{ role = "user"; content = "Reply with only the word OK." })
  max_tokens = 16
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://localhost:3000/v1/chat/completions -ContentType "application/json" -Body $body
```

## Troubleshooting Checklist
- **Worker not discovered:** Check Windows Firewall allows `TCP 50052` on the worker (`Private,Domain`), confirm both machines are on the same physical LAN subnet, or set `"workers": ["<WORKER_LAN_IP>:50052"]` in the Coordinator's `ilima.json` and call `POST http://localhost:3000/api/cluster/reload`.
- **RPC protocol / crash on model load:** Verify both Coordinator and Worker run the exact same `llama.cpp` build (`b12503`).
- **Slow prefill or decode over Ethernet:** Verify the link negotiated at `1000 Mbps` (Gigabit), not `100 Mbps` (`Get-NetAdapter | Select Name, LinkSpeed`), and prefer wired Ethernet over Wi-Fi.
- **Worker fell back to CPU/iGPU:** Run `& .\runtime\llama-server.exe --list-devices` on the worker and confirm the discrete GPU appears (`CUDA0` or `Vulkan0`).
