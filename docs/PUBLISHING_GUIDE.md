# HK Package Publishing & Distribution Guide

This guide details how to publish and synchronize all **HK Neural Tensor Framework** packages across global registries, following the convention: **`hk` wherever possible, `hknt` elsewhere**:

| Package Name | Ecosystem | Registry | Configuration | Package Type |
|:---|:---|:---|:---|:---|
| **`hk`** | Python / PyTorch | **[PyPI](https://pypi.org/project/hk/)** | [`pyproject.toml`](../pyproject.toml) | Wheel (`.whl`) & Sdist (`.tar.gz`) |
| **`hknt`** | TypeScript / JS | **[npm](https://www.npmjs.com/package/hknt)** | [`bindings/js/package.json`](../bindings/js/package.json) | npm Tarball (`.tgz`) |
| **`hknt`** | Rust | **[crates.io](https://crates.io/crates/hknt)** | [`bindings/rust/Cargo.toml`](../bindings/rust/Cargo.toml) | Cargo Crate (`.crate`) |
| **`Hk`** | C# / .NET | **[NuGet](https://www.nuget.org/packages/Hk)** | [`bindings/csharp/Hk.csproj`](../bindings/csharp/Hk.csproj) | NuGet Package (`.nupkg`) |
| **`hk/bindings/go`** | Go | **GitHub / pkg.go.dev** | [`bindings/go/go.mod`](../bindings/go/go.mod) | Go Module |
| **Native Binaries** | C / C++ / CLI | **[GitHub Releases](https://github.com/harshitkhandelwal208/hk/releases)** | [`build.zig`](../build.zig) | `.exe`, `.dll`, `.so`, `.dylib` |

---

## 1. Automated Release via GitHub Actions

The repository includes a release pipeline in [`.github/workflows/release.yml`](../.github/workflows/release.yml).

### Step 1: Set Repository Secrets (One-time)
In your GitHub repo settings (`Settings` > `Secrets and variables` > `Actions`):
- `PYPI_API_TOKEN`: PyPI token with upload permissions (or configure PyPI Trusted Publishing / OIDC).
- `NPM_TOKEN`: npm granular access token with Read and Write permissions for `hknt`.
- `CARGO_REGISTRY_TOKEN`: crates.io API token with publish permissions for `hknt`.
- `NUGET_API_KEY`: nuget.org API key with push permissions for `Hk`.

### Step 2: Trigger the Release
Push a git version tag:
```bash
git tag -a v1.0.0 -m "Release v1.0.0: HK Neural Tensor Framework"
git push origin v1.0.0
```

GitHub Actions will automatically:
1. Cross-compile native Zig binaries (`x86_64-linux`, `aarch64-linux`, `x86_64-windows`, `aarch64-macos`).
2. Build multiplatform Python wheels and sdist.
3. Create a GitHub Release with all compiled assets attached.
4. Publish `hk` to PyPI.
5. Publish `hknt` to npm.
6. Publish `hknt` to crates.io.
7. Publish `Hk` to NuGet.

---

## 2. Direct Local Publishing

All packages can also be published directly from your local terminal:

### A. Python (`hk` on PyPI)
1. Build the distribution:
   ```bash
   py -3.12 -m build
   ```
2. Verify package metadata:
   ```bash
   py -3.12 -m twine check dist/*
   ```
3. Upload to PyPI:
   ```bash
   py -3.12 -m twine upload dist/*
   ```

---

### B. TypeScript / Node.js (`hknt` on npm)
1. Build and package:
   ```bash
   cd bindings/js
   npm run build
   npm pack
   ```
2. Authenticate and publish:
   ```bash
   npm login
   npm publish --access public
   ```

---

### C. Rust (`hknt` on crates.io)
1. Package the crate:
   ```bash
   cd bindings/rust
   cargo package
   ```
2. Authenticate and publish:
   ```bash
   cargo login <your-crates-io-token>
   cargo publish
   ```

---

### D. C# / .NET (`Hk` on NuGet)
1. Build and pack the `.nupkg`:
   ```bash
   cd bindings/csharp
   dotnet pack -c Release
   ```
2. Push to NuGet:
   ```bash
   dotnet nuget push bin/Release/Hk.1.0.0.nupkg --api-key <your-nuget-key> --source https://api.nuget.org/v3/index.json
   ```

---

### E. Go Module
Go modules are fetched directly from GitHub tags:
```bash
git tag bindings/go/v1.0.0
git push origin bindings/go/v1.0.0
```
Consumers can then import:
```go
import "github.com/harshitkhandelwal208/hk/bindings/go/hk"
```
