# Security Policy

The HK team takes the security and integrity of our codebase, binary containers, and parser interfaces seriously.

---

## Supported Versions

Only the latest stable release line is actively supported with security patches and critical bug fixes.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

If you discover a security vulnerability in the HK binary format, Zig runtime, C API bindings, or Python deserialization logic (e.g. out-of-bounds memory read, buffer overflow, arbitrary code execution, or container denial-of-service), **please do not open a public issue.**

Instead, please report it through one of the following channels:
1. **GitHub Private Vulnerability Reporting**: Go to the **Security** tab of the HK repository and click **"Report a vulnerability"**.
2. **Email**: Send details directly to `security@hk-ai.org` (or the repository maintainers).

### What to Include in Your Report
- A description of the vulnerability and its potential impact.
- Clear step-by-step reproduction steps or a minimal proof-of-concept (`.hk` or `.gguf` file).
- The operating system, architecture, and version of HK being run.

### Response Timeline
- **Initial Acknowledgement**: Within 48 hours.
- **Triage & Reproduction**: Within 5 business days.
- **Fix & Advisory**: We will coordinate a release date and credit your contribution in release notes and CVE filings (unless you prefer anonymity).
