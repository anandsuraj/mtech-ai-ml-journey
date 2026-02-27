#!/usr/bin/env bash
set -euo pipefail

RELEASE="heart-prod"

echo "[INFO] Available revisions:"
helm history "$RELEASE"

echo "[INFO] Rolling back to previous revision"
helm rollback "$RELEASE" || {
  echo "[ERROR] Rollback failed"
  exit 1
}

echo "[SUCCESS] Rollback completed"
kubectl get pods
