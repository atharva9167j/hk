#!/usr/bin/env bash
# HK Universal AI Framework - Web Installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/harshitkhandelwal208/hk/main/install.sh | bash
set -e

REPO="harshitkhandelwal208/hk"
VERSION="v1.0.0"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     PLATFORM="linux";;
    Darwin*)    PLATFORM="macos";;
    *)          echo "Error: Unsupported operating system: $OS"; exit 1;;
esac

# Detect Architecture
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)   ARCH_NAME="x86_64";;
    aarch64|arm64)  ARCH_NAME="aarch64";;
    *)              echo "Error: Unsupported architecture: $ARCH"; exit 1;;
esac

if [ "$PLATFORM" = "macos" ] && [ "$ARCH_NAME" = "aarch64" ]; then
    BINARY_NAME="hk-macos-arm64"
elif [ "$PLATFORM" = "macos" ]; then
    BINARY_NAME="hk-macos-x86_64"
elif [ "$PLATFORM" = "linux" ] && [ "$ARCH_NAME" = "aarch64" ]; then
    BINARY_NAME="hk-linux-aarch64"
else
    BINARY_NAME="hk-linux-x86_64"
fi

DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY_NAME}"
INSTALL_DIR="/usr/local/bin"

if [ ! -w "$INSTALL_DIR" ]; then
    INSTALL_DIR="${HOME}/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

TARGET="${INSTALL_DIR}/hk"

echo "==> Downloading HK CLI (${VERSION}) for ${PLATFORM}-${ARCH_NAME}..."
curl -fsSL "$DOWNLOAD_URL" -o "$TARGET"
chmod +x "$TARGET"

echo "==> Successfully installed HK CLI to: $TARGET"
if ! command -v hk >/dev/null 2>&1; then
    echo "Note: Ensure ${INSTALL_DIR} is in your PATH."
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi
echo "==> Run 'hk --help' to get started."
