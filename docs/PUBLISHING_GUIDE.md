# HK Package Publishing & GitHub Synchronization Guide

This guide explains how to publish and synchronize the **HK Neural Tensor Framework** packages across **GitHub**, **PyPI (Python / `pip`)**, and **npm (TypeScript / JavaScript)**.

---

## 1. Architecture of HK Packages

The repository is organized to distribute packages across language ecosystems from a single source of truth:

| Package | Ecosystem | Source Directory | Configuration File | Target Registry |
| :--- | :--- | :--- | :--- | :--- |
| **`hk`** (or `hk-tensor`) | Python / PyTorch | Repository root | [`pyproject.toml`](../pyproject.toml) | [PyPI](https://pypi.org) |
| **`@hk-format/core`** (or `@harshitkhandelwal208/hk`) | Node.js / Browser | [`bindings/js/`](../bindings/js) | [`bindings/js/package.json`](../bindings/js/package.json) | [npm](https://npmjs.com) |
| **Native Binaries** (`.dll`, `.so`, `.dylib`) | C/C++/Zig | [`src/`](../src) | [`build.zig`](../build.zig) | [GitHub Releases](https://github.com/harshitkhandelwal208/hk/releases) |

---

## 2. Synchronized GitHub Actions Release (Automated)

The repository includes a unified workflow: [`.github/workflows/release.yml`](../.github/workflows/release.yml).  
Whenever you push a version tag (e.g. `v1.0.0`), GitHub Actions will:
1. Compile native Zig shared libraries across 4 targets (`x86_64-linux`, `aarch64-linux`, `x86_64-windows`, `aarch64-macos`).
2. Build Python wheels and source distribution (`dist/*.whl`, `dist/*.tar.gz`).
3. Create a GitHub Release and attach all native binary assets.
4. Publish the Python wheel to **PyPI** (`pip install hk`).
5. Compile TypeScript and publish to **npm** (`npm install @hk-format/core`).

### One-Time Setup on GitHub

To enable automated publishing, configure the following secrets and settings in your GitHub repository:

#### Step A: NPM Setup
1. Log in to [npmjs.com](https://www.npmjs.com).
2. Go to **Access Tokens** > **Generate New Token** > choose **Granular Access Token** (or Classic Automation token).
3. Set permissions to **Read and Write** for packages.
4. On GitHub, navigate to:  
   `https://github.com/harshitkhandelwal208/hk/settings/secrets/actions`
5. Click **New repository secret**:
   - Name: `NPM_TOKEN`
   - Value: `<your-npm-token>`

#### Step B: PyPI Setup (Trusted Publisher - Recommended)
PyPI supports **Trusted Publishing (OIDC)**, which eliminates the need to store long-lived passwords or API tokens:
1. Log in to [pypi.org](https://pypi.org).
2. Go to **Account Settings** > **Publishing**.
3. Under **Add a new publisher**:
   - PyPI Project Name: `hk` (or `hk-tensor` if using that name)
   - Owner: `harshitkhandelwal208`
   - Repository: `hk`
   - Workflow name: `release.yml`
   - Environment name: (leave blank or enter `pypi`)
4. *Alternative (API Token)*: If preferred, create an API token on PyPI and add it to GitHub secrets as `PYPI_API_TOKEN`.

---

## 3. How to Trigger a Synchronized Release

Whenever you are ready to publish a new version:

1. **Update versions** in the repository:
   - In `pyproject.toml`: update `version = "1.0.0"`
   - In `bindings/js/package.json`: update `"version": "1.0.0"`
   - In `python/hk/__init__.py`: update `__version__ = "1.0.0"`
   - In `CHANGELOG.md`: document the release highlights

2. **Commit and push to main**:
   ```bash
   git add .
   git commit -m "chore: bump version to 1.0.0"
   git push origin main
   ```

3. **Tag the release and push the tag**:
   ```bash
   git tag -a v1.0.0 -m "HK Framework v1.0.0 Release"
   git push origin v1.0.0
   ```

4. Watch the progress in the **Actions** tab:  
   `https://github.com/harshitkhandelwal208/hk/actions`

---

## 4. Manual / Local Publishing

If you ever want to publish locally without waiting for GitHub Actions, follow these steps:

### A. Publishing to PyPI (Python / `pip`)

1. **Install build and upload tools**:
   ```bash
   pip install --upgrade build twine
   ```

2. **Build the source distribution and binary wheel**:
   ```bash
   # Run from the root of the repository
   python -m build
   ```
   This produces files in `dist/`, e.g.:
   - `dist/hk-1.0.0.tar.gz`
   - `dist/hk-1.0.0-py3-none-any.whl`

3. **Check package integrity**:
   ```bash
   python -m twine check dist/*
   ```

4. **Upload to PyPI**:
   ```bash
   # Test on TestPyPI first (optional):
   python -m twine upload --repository testpypi dist/*

   # Publish to production PyPI:
   python -m twine upload dist/*
   ```
   *(Enter your PyPI `__token__` username and API token password when prompted).*

---

### B. Publishing to npm (TypeScript / JavaScript)

1. **Navigate to the JS binding directory**:
   ```bash
   cd bindings/js
   ```

2. **Install dependencies and compile TypeScript**:
   ```bash
   npm install
   npm run build
   ```
   This compiles `hk.ts` to `dist/hk.js` and `dist/hk.d.ts`.

3. **Verify the package payload**:
   ```bash
   npm pack --dry-run
   ```
   Ensure only `dist/`, `README.md`, and `package.json` are included.

4. **Log in to npm**:
   ```bash
   npm login
   ```

5. **Publish to npm**:
   ```bash
   # For scoped packages (e.g. @hk-format/core or @harshitkhandelwal208/hk):
   npm publish --access public

   # For unscoped packages (e.g. hk-tensor):
   npm publish
   ```

---

## 5. Package Naming Notes & Recommendations

### PyPI (`pip`)
- **`hk`**: Two-letter names on PyPI may be protected or require PyPI administrator review. If `hk` is reserved or blocked during upload, change `name = "hk-tensor"` or `name = "hk-format"` in `pyproject.toml`. Python imports remain `import hk` regardless of whether the pip package is named `hk` or `hk-tensor`.

### npm
- **`@harshitkhandelwal208/hk`**: Guaranteed free, namespaced under your GitHub/npm username, no name collisions possible.
- **`@hk-format/core`**: Requires creating an organization named `hk-format` on npm (free for public packages).
- **`hk-tensor`**: Unscoped, available for direct registration.

To change the npm package name, edit the `"name"` field in [`bindings/js/package.json`](../bindings/js/package.json).

---

## 6. Git Remote Configuration

Both SSH and HTTPS are supported for this repository:

- **HTTPS with GitHub CLI** (Currently active):
  ```bash
  git remote set-url origin https://github.com/harshitkhandelwal208/hk.git
  ```
- **SSH** (Requires uploading your `~/.ssh/id_ed25519.pub` to GitHub Settings > SSH Keys):
  ```bash
  git remote set-url origin git@github.com:harshitkhandelwal208/hk.git
  ```
