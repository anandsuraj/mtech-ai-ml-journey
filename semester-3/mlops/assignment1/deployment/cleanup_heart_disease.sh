#!/usr/bin/env bash
set -euo pipefail

RELEASE="heart-prod"

echo "[INFO] Uninstalling Helm release"
helm uninstall "$RELEASE" || echo "[WARN] Release not found"

echo "[INFO] Stopping Minikube"
minikube stop || true

echo "[SUCCESS] Cleanup completed"