# Security Policy

## Network Trust Model
**Ilima is designed for trusted private LANs.** The Coordinator HTTP API (`:3000`), managed `llama-server` (`:8081`), and Worker RPC port (`:50052`) do not implement authentication and must **not** be exposed directly to the public internet. Keep Windows/OS firewall rules scoped to `Private,Domain` profiles only.

## Reporting a Vulnerability
Please **do not** report security vulnerabilities in public GitHub issues.

Instead, report them privately via:
- **GitHub Private Vulnerability Reporting** (under the repository's **Security** tab -> *Report a vulnerability*), or
- Email: `mk@hitprotocol.org`

Include a clear description of the issue and steps to reproduce. You can expect an acknowledgement within a few days.
