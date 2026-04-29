#!/bin/bash
# Configure K3s to use NVIDIA GPU on Jetson Orin (JetPack 6.x)
# Run this BEFORE installing K3s, or re-install K3s after running.
#
# Requires: JetPack 6.x with nvidia-container-toolkit installed.

set -e

echo "==> Checking prerequisites..."

if ! command -v nvidia-container-runtime &>/dev/null; then
    echo "ERROR: nvidia-container-runtime not found."
    echo "Please ensure JetPack is installed before running this script."
    exit 1
fi

echo "  nvidia-container-runtime: OK"

# Verify system containerd has NVIDIA runtime configured AND set as default.
# The --set-as-default flag is critical: without it, pods that request
# `resources.limits.nvidia.com/gpu: 1` will land on runc (no GPU) unless
# they also set `runtimeClassName: nvidia` explicitly.
NEED_CONTAINERD_RESTART=0
if ! grep -q 'default_runtime_name = "nvidia"' /etc/containerd/config.toml 2>/dev/null; then
    echo "  System containerd missing or not defaulting to NVIDIA. Configuring now..."
    sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
    NEED_CONTAINERD_RESTART=1
else
    echo "  System containerd NVIDIA default runtime: OK"
fi

# System containerd does not know where K3s drops its CNI binaries and
# configs, so pods never become Ready ("cni plugin not initialized").
# Patch the [plugins."io.containerd.grpc.v1.cri".cni] section to point at
# K3s's internal paths. Idempotent.
if ! grep -q 'bin_dir = "/var/lib/rancher/k3s' /etc/containerd/config.toml 2>/dev/null; then
    echo "  Patching containerd CNI paths for K3s..."
    sudo python3 - <<'PYEOF'
# Idempotent CNI block patcher.
# Two cases on Jetson:
#   1) Fresh JetPack: containerd config has no [..."cri".cni] block at all.
#      ⇒ Insert ours after the [..."cri"] anchor.
#   2) Pre-existing config (e.g. from earlier nvidia-ctk run, or distro default):
#      already has a [..."cri".cni] block with default paths (e.g. /opt/cni/bin).
#      ⇒ Strip the existing block FIRST so we don't end up with a duplicate
#         table (containerd would refuse to load the file).
import re
p = '/etc/containerd/config.toml'
with open(p) as f: lines = f.readlines()

cni_header = '[plugins."io.containerd.grpc.v1.cri".cni]'
cri_anchor = '[plugins."io.containerd.grpc.v1.cri"]'

# Pass 1: drop any existing cni block (header + body until next section / EOF).
out, in_cni = [], False
for line in lines:
    s = line.lstrip()
    if s.startswith(cni_header):
        in_cni = True
        continue
    if in_cni:
        # leave the block when we hit the next [section] (but not [[arrays]])
        if s.startswith('[') and not s.startswith('[['):
            in_cni = False
        else:
            continue
    out.append(line)

# Pass 2: insert our block right after [..."cri"] anchor.
new, inserted = [], False
for line in out:
    new.append(line)
    if not inserted and line.lstrip().startswith(cri_anchor) \
       and not line.lstrip().startswith(cri_anchor + '.'):
        new += [
            '    [plugins."io.containerd.grpc.v1.cri".cni]\n',
            '      bin_dir = "/var/lib/rancher/k3s/data/current/bin"\n',
            '      conf_dir = "/var/lib/rancher/k3s/agent/etc/cni/net.d"\n',
        ]
        inserted = True

if inserted:
    with open(p, 'w') as f: f.writelines(new)
PYEOF
    NEED_CONTAINERD_RESTART=1
else
    echo "  containerd CNI paths: OK"
fi

if [ "$NEED_CONTAINERD_RESTART" = "1" ]; then
    sudo systemctl restart containerd
    echo "  containerd restarted."
fi

echo ""
echo "==> Installing K3s using system containerd..."
echo "    (Using /run/containerd/containerd.sock so K3s inherits NVIDIA runtime)"
echo ""

curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server \
  --container-runtime-endpoint unix:///run/containerd/containerd.sock \
  --kubelet-arg=container-log-max-files=5 \
  --kubelet-arg=container-log-max-size=10Mi" sh -

echo ""
echo "==> Waiting for K3s to be ready..."
sleep 10
sudo k3s kubectl wait --for=condition=Ready node --all --timeout=60s

echo ""
echo "==> Setting up kubectl access..."
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$USER:$USER" ~/.kube/config
echo "    KUBECONFIG written to ~/.kube/config"

echo ""
echo "==> Done! Verify with:"
echo "    kubectl get nodes"
echo "    kubectl describe node | grep -A5 'Capacity:'"
