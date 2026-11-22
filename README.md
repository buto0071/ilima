# Ilima

> *Ilima* (isiZulu / isiXhosa: a communal work party pooling effort for a heavy task) — Why buy another GPU when you have another computer? A lightweight Rust orchestrator for running `llama.cpp` across heterogeneous machines on your LAN.

## What it does
- **Heterogeneous layer-splitting via `llama.cpp` RPC:** Pool GPUs across different vendors and machines (e.g., AMD Vulkan desktop + NVIDIA CUDA laptop) over LAN.
- **Automatic VRAM placement:** Probes local and remote free VRAM, keeps small models 100% local, and computes `--tensor-split` automatically (including `1,950 MB` extra Coordinator headroom when `--spec-type draft-mtp` is enabled).
- **Zero-config runtime & LAN discovery:** Auto-downloads pinned `llama.cpp` `b12503` binaries into `./runtime` if missing, discovers workers via mDNS + `/24` LAN sweep, ranks links by live TCP RTT (preferring Gigabit Ethernet over Wi-Fi), and hot-recruits workers as they join.
- **RPC prefill fix:** Automatically injects `--ctx-checkpoints 0` when speculative decoding (`spec_type`) is active so hybrid recurrent models do not stall during RPC prompt evaluation.
- **OpenAI-compatible API + Web Dashboard:** Serves `/v1/models`, `/v1/chat/completions` (with automatic model switching), `/api/*`, a React control panel on port `3000`, and a one-liner Windows worker join endpoint (`GET /join`).

## Architecture
```
 Coordinator                                  Worker
 Desktop — AMD RX 9070 XT 16 GB (Vulkan)      Laptop — NVIDIA RTX 5050 8 GB (CUDA)
 ilima + llama-server        <--- 1 GbE --->  ilima worker + ggml-rpc-server
   :3000 (API / UI / /join)                     :50052 (RPC)
   :8081 (llama-server)
```

## Benchmarks (`Qwen3.8-27B-Q4_K_M`, 64k Context)

Measured on **Desktop Coordinator (AMD Radeon RX 9070 XT 16 GB)** + **Laptop Worker (NVIDIA GeForce RTX 5050 Laptop 8 GB)** running `Qwen3.8-27B-Q4_K_M.gguf` (~15.66 GB on disk) at `-c 64192 -ngl 99 -fa on` (native `f16` KV cache).

| Setup | Runtime | Network & Split (`RPC0,Local`) | Prompt Eval (`pp`) | Decode Avg (`tg`) | Decode Peak (`tg_3s`) | Speedup vs Local |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Local Only (16 GB GPU + PCIe RAM spill)** | Vulkan `b12038` | Single GPU (no Ilima) | `84.29 tok/s` | `2.65 tok/s` | `2.83 tok/s` | `1.00×` |
| **Ilima RPC (Wi-Fi)** | Vulkan `b12038` | Wi-Fi, `--tensor-split 42,58` | `11.41 tok/s` | `9.95 tok/s` | `10.19 tok/s` | `3.75×` |
| **Ilima RPC (1 GbE + ROCm)** | ROCm `b12073` | Ethernet, `--tensor-split 22,78` | `55.93 tok/s` | `19.19 tok/s` | `19.79 tok/s` | `7.24×` |
| **Ilima RPC (1 GbE + Vulkan, No MTP)** | Vulkan `b12038` | Ethernet, `--tensor-split 22,78` / `20,80` | `55.15–62.66 tok/s` | `25.03 tok/s` | `25.68–26.01 tok/s` | `9.45×` |
| **Ilima RPC + Built-in MTP (General prose — 54 Presidents)** | `b12503` | 1 Gbps GbE, split `32,68`, `draft-mtp` (`n=2`) | **`197.05 tok/s`** | **`40.00 tok/s`** | **`44.27 tok/s`** | **`15.09×`** |
| **Ilima RPC + Built-in MTP (Code gen — Python BST, `medium`)** | `b12503` | 1 Gbps GbE, split `32,68`, `draft-mtp` (`n=2`) | **`86.32 tok/s`** *(short prompt)* | **`42.99 tok/s`** | **`44.76 tok/s`** | **`16.22×`** |
| **Ilima RPC + Built-in MTP (5,032-token prompt ingestion)** | `b12503` | 1 Gbps GbE, split `32,68`, `draft-mtp` (`n=2`) | **`789.39 tok/s`** (`903.93` peak) | `30.05 tok/s` *(@ 5.3k ctx)* | `32.88 tok/s` | — |

**`Q4_K_M` vs. smaller quants (`Ridge-3.7bpw`):** On the same 64k Python BST coding test with built-in `blk.64` MTP (`b12503`, recorded on a 100 Mbps link before reseating to 1 Gbps), full `Qwen3.8-27B-Q4_K_M` (15.66 GB, split `32,68`) reached `39.24–40.60 tok/s` (`medium` effort) / `41.16 tok/s` (`low` effort, `44.54` peak, `87.41 tok/s` prefill), outperforming `Qwen3.8-27B-Ridge-3.7bpw` (11.73 GB) both locally on one GPU (`32.12 tok/s` without MTP at 32k; `16.56 tok/s` with MTP at 16k) and clustered at 64k (`36.81 tok/s` avg, `73.70 tok/s` prefill).

## Quickstart

> **Using a coding agent to set this up?** Point it at [`AGENTS.md`](AGENTS.md) for a step-by-step setup and verification playbook.

### Prerequisites
- **Rust (stable)** toolchain (`cargo`).
- **Node.js 20+** (optional — only needed to build the web dashboard).
- **Wired Gigabit Ethernet** between Coordinator and Worker is strongly recommended (Wi-Fi and 100 Mbps links bottleneck RPC traffic).
- **`llama.cpp` `b12503`:** Downloaded automatically into `./runtime` on first run (Vulkan on Windows/Linux, Metal on macOS). All nodes in a cluster **must** run the exact same `llama.cpp` build tag (`b12503`) so the RPC wire protocol matches.
  - *Tip for NVIDIA workers:* For maximum worker speed, download the official `llama-b12503-bin-win-cuda-*` release from `ggml-org/llama.cpp` into your `Downloads` folder (or point `llama_bin_dir` at it). The Windows `/join` script detects it automatically; otherwise Vulkan `b12503` is synced from the Coordinator and works out of the box.

### 1. Build on the Coordinator (main machine)
```powershell
cargo build --release -p ilima

# Optional: build the web dashboard UI
cd ilima-dashboard
npm ci
npm run build
cd ..
```

### 2. Configure the Coordinator
```powershell
Copy-Item ilima.example.json ilima.json
New-Item -ItemType Directory -Force models | Out-Null
```
Place one or more `.gguf` model files in `models/` (or add extra directories to `models_dirs` / set `model_path` in `ilima.json`).

**Optional — Built-in MTP (`Qwen3.8-27B-Q4_K_M` and other models with embedded `blk.64` MTP weights):**
Set these fields in `ilima.json`:
```json
"ctx_size": 64192,
"spec_type": "draft-mtp",
"draft_max": 2,
"draft_min": 0,
"reasoning_effort": "medium",
"extra_server_args": ["-fa", "on"]
```
*(Leave `"spec_type": null` for standard models without a built-in MTP head.)*

### 3. Start the Coordinator
```powershell
.\target\release\ilima.exe start --config ilima.json
```
On first launch, if `./runtime/llama-server.exe` does not exist yet, Ilima automatically downloads and extracts `llama.cpp` `b12503` into `./runtime`. Open the dashboard at `http://localhost:3000`.

### 4. Add a Worker machine
**Option A — Windows one-liner (fastest):**
On the second Windows PC on the same LAN, open PowerShell and run (replace `<COORDINATOR_LAN_IP>` with the Coordinator's `192.168.x.x` / `10.x.x.x` IP):
```powershell
irm http://<COORDINATOR_LAN_IP>:3000/join | iex
```
This downloads `ilima.exe` and the matching `b12503` runtime from the Coordinator, adds Private/Domain Windows Firewall rules, starts the GPU `rpc-server` on port `50052`, and notifies the Coordinator to reload.

**Option B — Manual worker setup (any OS):**
Copy `ilima.worker.example.json` to `ilima.json` on the worker, ensure `llama.cpp` `b12503` is in `./runtime` (or set `llama_bin_dir`), and run:
```powershell
.\target\release\ilima.exe worker --config ilima.json
```
*(If auto-discovery is blocked by a firewall, add `"workers": ["<WORKER_LAN_IP>:50052"]` to the Coordinator's `ilima.json`.)*

### 5. Use via Dashboard or OpenAI-compatible API
- **Dashboard:** `http://localhost:3000`
- **OpenAI API base URL:** `http://localhost:3000/v1` (works with OpenCode, Continue, Open WebUI, curl, etc. — requesting any discovered model name in `POST /v1/chat/completions` loads/switches to it automatically).
- **Quick health checks:**
  - `GET http://localhost:3000/api/status`
  - `GET http://localhost:3000/api/cluster`
  - `GET http://localhost:3000/api/peers`
  - `GET http://localhost:3000/v1/models`

## Known Limitations
- `llama.cpp` RPC splits **layers over the network** (this is not unified memory); link bandwidth and latency directly impact speed — use wired Gigabit Ethernet, not Wi-Fi.
- Multi-slot `--parallel 2` together with `draft-mtp` is currently held back pending upstream `llama.cpp` state-isolation fixes (keep `"parallel_slots": 1` when using `draft-mtp`).
- Developed and tested primarily on Windows (AMD Vulkan Coordinator + NVIDIA CUDA/Vulkan Worker). Linux and macOS support in `runtime_manager.rs` is lightly tested.
- Dashboard frontend currently targets `http://localhost:3000` (API) and `http://localhost:8081/slots` (live token telemetry) when viewed from the Coordinator host.

## License
MIT
