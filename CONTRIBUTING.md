# Contributing to Ilima

Thanks for your interest in contributing to **Ilima**!

## Guidelines
- **Open an issue first** before starting large architectural changes so we can align on scope.
- **Keep PRs focused:** One logical fix or feature per pull request.
- **Never commit local artifacts or personal info:** Do not commit `ilima.json`, `.gguf`/`.bin` model weights, downloaded `llama.cpp` binaries, personal file paths, or real LAN/VPN IPs.
- **Preserve the `llama.cpp` RPC version invariant:** Coordinator and Worker nodes must stay aligned on the pinned `llama.cpp` release tag unless an upgrade explicitly updates both sides and documentation together.

## Before Opening a Pull Request
Run these checks locally from the repository root:
```powershell
cargo build --workspace
cargo test --workspace
```
If you modified `ilima-dashboard/`:
```powershell
cd ilima-dashboard
npm ci
npm run build
```

All pull requests targeting `main` require passing CI checks (`Rust Build & Test` and `Dashboard Build`) and maintainer review (`@buto0071`).
