#!/usr/bin/env bash
set -euo pipefail

### ---------- CONFIG ----------
REPO_URL="https://github.com/KSharma-SourceCode/AI-ML-Assignments.git"
DEPLOY_DIR="AI-ML-Assignments/deployment/helm"
HELM_RELEASE="heart-prod"
CHART_NAME="heart-disease"
MINIKUBE_DRIVER="docker"
### ----------------------------

log() {
  echo -e "\n[INFO] $1"
}

fail() {
  echo -e "\n[ERROR] $1"
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

### ---------- STEP 1: SYSTEM CHECK ----------
log "Checking OS requirements"

if [[ "$EUID" -eq 0 ]]; then
  fail "Do NOT run this script as root"
fi

### ---------- STEP 2: INSTALL DOCKER ----------
if ! command_exists docker; then
  log "Installing Docker"
  sudo apt update
  sudo apt install -y docker.io || fail "Docker install failed"
  sudo usermod -aG docker "$USER"
  log "Docker installed. Re-login required after script completes."
else
  log "Docker already installed"
fi

### ---------- STEP 3: INSTALL KUBECTL ----------
if ! command_exists kubectl; then
  log "Installing kubectl"
  curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl \
    || fail "kubectl download failed"
  chmod +x kubectl
  sudo mv kubectl /usr/local/bin/ || fail "kubectl install failed"
else
  log "kubectl already installed"
fi

### ---------- STEP 4: INSTALL MINIKUBE ----------
if ! command_exists minikube; then
  log "Installing Minikube"
  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
    || fail "Minikube download failed"
  sudo install minikube-linux-amd64 /usr/local/bin/minikube \
    || fail "Minikube install failed"
else
  log "Minikube already installed"
fi

### ---------- STEP 5: INSTALL HELM ----------
if ! command_exists helm; then
  log "Installing Helm"
  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash \
    || fail "Helm install failed"
else
  log "Helm already installed"
fi

### ---------- STEP 6: START MINIKUBE ----------
log "Starting Minikube"
minikube start --driver="$MINIKUBE_DRIVER" || fail "Minikube failed to start"

### ---------- STEP 7: ENABLE INGRESS ----------
log "Enabling Ingress controller"
minikube addons enable ingress || fail "Ingress enable failed"

### ---------- STEP 8: CLONE DEPLOYMENT FILES ----------
log "Cloning deployment repository"
rm -rf AI-ML-Assignments
git clone "$REPO_URL" || fail "Git clone failed"

cd "$DEPLOY_DIR" || fail "Deployment directory not found"

### ---------- STEP 9: HELM DEPLOY ----------
log "Deploying application using Helm"

if helm status "$HELM_RELEASE" >/dev/null 2>&1; then
  log "Existing release found. Upgrading..."
  helm upgrade "$HELM_RELEASE" "$CHART_NAME" || fail "Helm upgrade failed"
else
  log "Installing new release"
  helm install "$HELM_RELEASE" "$CHART_NAME" || fail "Helm install failed"
fi

### ---------- STEP 10: VERIFY ----------
log "Verifying Kubernetes resources"
kubectl get pods || fail "Pods not running"
kubectl get svc || fail "Service not created"
kubectl get ingress || fail "Ingress not created"

### ---------- STEP 11: ACCESS INFO ----------
SERVICE_URL=$(minikube service "$HELM_RELEASE" --url | head -n 1)

log "Deployment SUCCESSFUL 🎉"
log "FastAPI Swagger UI:"
echo "👉 $SERVICE_URL/docs"

log "NOTE: Keep terminal open if Minikube service tunnel is active"
