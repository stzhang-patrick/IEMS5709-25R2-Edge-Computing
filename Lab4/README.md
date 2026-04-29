# Lab 4: Multi-Service AI Deployment with K3s

## Learning Objectives

By the end of this lab, you will be able to:

- Install and configure a single-node K3s cluster on Jetson Orin
- Use `kubectl` to inspect, deploy, and manage workloads
- Understand the five core Kubernetes objects: Pod, Deployment, Service, ConfigMap, PersistentVolumeClaim
- Deploy GPU workloads in K3s with the NVIDIA device plugin
- Migrate a Docker Compose stack to Kubernetes manifests

## Environment Notes

This lab runs on **NVIDIA Jetson Orin NX** with JetPack 6.x. Steps that differ on an Ubuntu workstation are marked:

- `[Jetson]` — Jetson Orin specific
- `[Workstation]` — Ubuntu x86-64 only
- *(no tag)* — works on both

---

## Part 1: Installing K3s

### What is K3s?

K3s is a lightweight, production-grade Kubernetes distribution from Rancher (SUSE). It packages the full Kubernetes API into a single ~70 MB binary, making it ideal for edge devices.

| | Docker Compose | K3s |
|---|---|---|
| Scope | Single machine | Cluster (1 to many nodes) |
| Self-healing | No (manual restart) | Yes (Deployment reconciles pods) |
| Service discovery | Container names | Kubernetes DNS |
| Scaling | Manual (`replicas:`) | `kubectl scale` / HPA |
| Production use | Dev / small deployments | Edge production |

### 1.1 Install K3s

**`[Workstation]`** — Basic install (no GPU):

```bash
curl -sfL https://get.k3s.io | sh -
```

**`[Jetson]`** — Install using system containerd so K3s inherits the NVIDIA runtime that JetPack already configured:

```bash
# JetPack's containerd already has the NVIDIA runtime set up.
# Tell K3s to use it via --container-runtime-endpoint.
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server \
  --container-runtime-endpoint unix:///run/containerd/containerd.sock" sh -
```

> Alternatively, run the helper script which also handles the NVIDIA device plugin:
> ```bash
> bash demo/02-gpu-workload/setup-gpu-runtime.sh
> ```

Wait ~30 seconds for K3s to start, then verify:

```bash
sudo systemctl status k3s
```

### 1.2 Configure kubectl Access

K3s writes a kubeconfig to `/etc/rancher/k3s/k3s.yaml`. Copy it to your home directory so you can run `kubectl` without `sudo`:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config

echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
source ~/.bashrc
```

### 1.3 Verify the Cluster

```bash
kubectl get nodes
```

Expected output:

```
NAME            STATUS   ROLES                  AGE   VERSION
jetson-orin-1   Ready    control-plane,master   2m    v1.30.x+k3s1
```

```bash
# View all pods K3s runs to manage itself
kubectl get pods -n kube-system
```

---

## Part 2: kubectl Basics

```mermaid
graph TD
    subgraph Cluster["K3s Cluster"]
        Node["Node<br>(Physical Device / VM)"]
        
        subgraph NS["Namespace<br>(Logical Resource Boundary)"]
            Deploy["Deployment<br>(Manages desired state & replicas)"]
            Svc["Service<br>(Stable Network IP / Load Balancer)"]
            
            Pod1["Pod 1<br>(Runs containers)"]
            Pod2["Pod 2<br>(Runs containers)"]
            
            Deploy -- "Creates & Monitors" --> Pod1
            Deploy -- "Creates & Monitors" --> Pod2
            
            Svc -- "Routes Traffic to" --> Pod1
            Svc -- "Routes Traffic to" --> Pod2
        end
        
        Pod1 -. "Scheduled on" .-> Node
        Pod2 -. "Scheduled on" .-> Node
    end
```

### Core Commands

**Cluster state**

```bash
kubectl get nodes
# NAME     STATUS   ROLES           AGE   VERSION
# durian   Ready    control-plane   12h   v1.34.6+k3s1
# - NAME: Hostname of the node
# - STATUS: Ready or NotReady
# - ROLES: control-plane or worker
# - AGE: How long the node has been running
# - VERSION: Version of Kubernetes + K3s version


kubectl get namespaces
# NAME              STATUS   AGE
# default           Active   12h
# kube-node-lease   Active   12h
# kube-public       Active   12h
# kube-system       Active   12h
# - NAME: Name of the namespace
# - STATUS: Active or Terminating
# - AGE: How long the namespace has been running
# - default: Default namespace (if you don't specify a namespace)
# - kube-node-lease: Namespace for node lease (heartbeat)
# - kube-public: Namespace for public resources
# - kube-system: Namespace for system resources


kubectl get pods                     # default namespace
# No resources found in default namespace.
# - Nothing deployed yet. The default namespace is empty until you run kubectl apply.

kubectl get pods -A                  # -A = --all-namespaces, shows every pod in the cluster
# NAMESPACE     NAME                                      READY   STATUS      RESTARTS   AGE
# kube-system   coredns-76c974cb66-pc8jc                  1/1     Running     0          12h
# kube-system   helm-install-traefik-crd-wvjk7            0/1     Completed   0          12h
# kube-system   helm-install-traefik-jfdj2                0/1     Completed   1          12h
# kube-system   local-path-provisioner-8686667995-zz94v   1/1     Running     0          12h
# kube-system   metrics-server-c8774f4f4-58fbs            1/1     Running     0          12h
# kube-system   svclb-traefik-5e30b8a8-f8449              2/2     Running     0          12h
# kube-system   traefik-c5c8bf4ff-sh9fj                   1/1     Running     0          12h
#
# Columns:
# - READY:    running containers / total (2/2 = both containers in the pod are up)
# - STATUS:   Running = active; Completed = job finished successfully and exited
# - RESTARTS: restart count (high values indicate the container keeps crashing)
#
# K3s system pods explained:
# - coredns:                DNS server — resolves "http://tts:8880" inside the cluster
# - helm-install-traefik-*: one-off install jobs (Completed is normal — they ran once and exited)
# - local-path-provisioner: creates PersistentVolumes automatically on the local disk
# - metrics-server:         collects CPU/memory data (used by kubectl top)
# - svclb-traefik:          load-balancer pods that forward traffic to the Traefik ingress
# - traefik:                ingress controller for routing HTTP into the cluster

kubectl get pods -n kube-system      # same as above, scoped to one namespace
# NAME                                      READY   STATUS      RESTARTS   AGE
# coredns-76c974cb66-pc8jc                  1/1     Running     0          12h
# ...
```

**Inspect a node**

```bash
kubectl describe node
# Name:    durian
# Roles:   control-plane
# ...
# Conditions:
#   Type            Status  Reason
#   MemoryPressure  False   KubeletHasSufficientMemory
#   DiskPressure    False   KubeletHasNoDiskPressure
#   PIDPressure     False   KubeletHasSufficientPID
#   Ready           True    KubeletReady               ← node is healthy
#
# Capacity:
#   cpu:     16                 ← total CPU cores on the machine
#   memory:  65530580Ki         ← total RAM (~64 GB on this workstation)
#   pods:    110                ← max pods this node can schedule
#
# On Jetson Orin after the GPU device plugin is deployed, you will also see:
#   nvidia.com/gpu: 4           ← single iGPU exposed as 4 schedulable replicas
#                                  via time-slicing (see Demo 2 / nvidia-device-plugin.yaml)
```

**Inspect a pod** *(requires a running pod — revisit after Demo 1)*

```bash
kubectl describe pod <pod-name>
# Shows: image, which node it runs on, resource requests/limits,
#        volume mounts, environment variables, and Events.
# The Events section at the bottom is the first place to look when a pod won't start:
#   Scheduled → Pulling → Pulled → Created → Started  (happy path)
#   Scheduled → Failed  (image pull error, insufficient resources, etc.)

kubectl logs <pod-name>              # stdout of the container
kubectl logs -f <pod-name>           # follow live output (like docker logs -f)
kubectl exec -it <pod-name> -- bash  # open a shell inside the running container
```

**Apply and delete resources**

```bash
kubectl apply -f manifest.yaml    # create or update resources declared in a YAML file
kubectl delete -f manifest.yaml   # delete those same resources
kubectl delete pod <pod-name>     # delete a specific pod (a Deployment will recreate it)
```

**Namespaces**

Kubernetes uses namespaces to isolate resources — like separate "projects" within one cluster. The four built-in ones are listed under the `kubectl get namespaces` output above. To create your own and switch into it:

```bash
kubectl create namespace lab4
kubectl get pods -n lab4

# Work in a namespace without typing -n every time:
kubectl config set-context --current --namespace=lab4
kubectl config set-context --current --namespace=default  # reset to default
```

**Useful output flags**

```bash
kubectl get pods -A -o wide
# NAMESPACE     NAME                    READY   STATUS    IP          NODE
# kube-system   coredns-76c974cb...     1/1     Running   10.42.0.3   durian
# ...
# -o wide adds: IP (the pod's in-cluster IP) and NODE (which machine runs this pod)
# Pod IPs (10.42.x.x) are internal — only reachable inside the cluster via Services.

kubectl get pods -o yaml             # dump the full YAML spec K3s is actually using
kubectl get pods --watch             # live view, refreshes on every change (Ctrl-C to stop)
```

**Debugging with events**

```bash
kubectl get events --sort-by='.lastTimestamp'
# LAST SEEN   TYPE      REASON     OBJECT              MESSAGE
# 54m         Normal    Scheduled  pod/hello-k3s-...   Successfully assigned to durian
# 54m         Normal    Pulling    pod/hello-k3s-...   Pulling image "python:3.12-slim"
# 54m         Normal    Pulled     pod/hello-k3s-...   Successfully pulled image in 20.991s
# 54m         Normal    Created    pod/hello-k3s-...   Created container: hello
# 54m         Normal    Started    pod/hello-k3s-...   Started container hello
#
# Events record every lifecycle change for every resource.
# TYPE:   Normal (expected) or Warning (something went wrong)
# REASON: short code — Scheduled, Pulling, Failed, OOMKilled, BackOff, ...
# This is the first command to run when a pod is stuck or crashing.
```

---

## Part 3: Core K3s Concepts

### 3.1 Pod

The smallest deployable unit. Wraps one (or more) containers that share network and storage.

```bash
# Run a one-off pod (like docker run, but in the cluster)
kubectl run test-pod --image=nginx:alpine --restart=Never
kubectl get pods
kubectl exec -it test-pod -- sh
kubectl delete pod test-pod
```

A pod YAML looks like this:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: my-container
    image: nginx:alpine
    ports:
    - containerPort: 80
```

**Key difference from Docker:** Pods are ephemeral. If a pod crashes, it is gone — no automatic restart. That is what Deployments are for.

### 3.2 Deployment

Manages a set of identical pods. Ensures the desired number of replicas is always running. If a pod crashes, the Deployment creates a new one automatically.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 2                       # keep 2 pods running at all times
  selector:
    matchLabels:
      app: my-app                   # which pods this Deployment owns
  template:                         # pod template (same structure as Pod spec)
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-container
        image: nginx:alpine
```

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deployment/my-app
kubectl scale deployment my-app --replicas=3
kubectl rollout undo deployment/my-app   # rollback to previous version
```

### 3.3 Service

Gives a stable network identity (DNS name + IP) to a set of pods. Pods come and go; the Service always points to the healthy ones.

| Service Type | Accessible From | Use Case |
|---|---|---|
| `ClusterIP` | Inside cluster only | Service-to-service calls |
| `NodePort` | Outside cluster via `<NodeIP>:<port>` | Dev / lab access |
| `LoadBalancer` | Via external LB | Cloud deployments |

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  type: NodePort
  selector:
    app: my-app           # routes to pods with this label
  ports:
  - port: 80              # ClusterIP port (used inside the cluster)
    targetPort: 80        # pod port
    nodePort: 30080       # external port (30000-32767)
```

**In-cluster DNS:** Any pod can reach `my-app` by hostname. K3s (like all K8s) provides automatic DNS:

```
http://my-app           → resolves within the same namespace
http://my-app.default   → from any namespace (namespace = default)
```

This is the K3s equivalent of Docker Compose service names like `http://vllm:8000`.

### 3.4 ConfigMap

Stores configuration data (files, environment variables) separately from the container image. Keeps images reusable.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
data:
  # Key-value pairs used as environment variables
  LOG_LEVEL: "info"
  MODEL_NAME: "Qwen3-4B"
  # File content injected as a file via volumeMount
  app.py: |
    print("Hello from ConfigMap!")
```

```yaml
# In a Deployment, inject as env vars:
env:
- name: LOG_LEVEL
  valueFrom:
    configMapKeyRef:
      name: my-config
      key: LOG_LEVEL

# Or mount as a file:
volumeMounts:
- name: config-vol
  mountPath: /app
volumes:
- name: config-vol
  configMap:
    name: my-config
```

### 3.5 PersistentVolumeClaim (PVC)

Requests persistent storage. The data survives pod restarts (unlike a plain volume).

Contrast with Docker Compose:

```yaml
# Docker Compose (Lab 3)
volumes:
  - open-webui-data:/app/backend/data

# K3s equivalent: first declare the claim
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: open-webui-data
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 2Gi
---
# Then reference it in the Deployment
volumes:
- name: data
  persistentVolumeClaim:
    claimName: open-webui-data
volumeMounts:
- name: data
  mountPath: /app/backend/data
```

K3s ships with a built-in `local-path` storage provisioner, so PVCs are automatically fulfilled with no extra setup.

```bash
kubectl get pvc
kubectl get pv
```

---

## Demo 1: Migrating hello-server to K3s

**Files:** `demo/01-hello-k3s/`

In Lab 3 we ran the hello-server with:

```bash
# Lab 3: Docker Compose
cd Lab3/demo/hello-server
docker compose build
docker compose up -d
curl http://localhost:5000 | jq
docker compose down
```

Now we will deploy the exact same Python server to K3s — without rebuilding the image — by injecting the source code via a ConfigMap.

### Step 1: Compare the structures

| Concept | Docker Compose (Lab 3) | K3s (Lab 4) |
|---|---|---|
| Code delivery | `COPY app.py .` in Dockerfile | ConfigMap `hello-k3s-app` |
| Run | `CMD ["python", "app.py"]` | Deployment `command:` |
| Port mapping | `ports: "5000:5000"` | Service NodePort 30500 |
| Image | Built locally (`build: .`) | `python:3.12-slim` (public) |

### Step 2: Apply the manifests

```bash
# ConfigMap holds the Python source code
kubectl apply -f demo/01-hello-k3s/configmap.yaml

# Deployment runs the code using the ConfigMap as a volume
kubectl apply -f demo/01-hello-k3s/deployment.yaml

# Service exposes it externally on port 30500
kubectl apply -f demo/01-hello-k3s/service.yaml
```

**How do these components connect?**

```mermaid
graph TD
    Svc["Service (hello-k3s-service)"]
    Deploy["Deployment (hello-k3s-deployment)"]
    CM["ConfigMap (hello-k3s-app)"]
    Pods["Pods with label app: hello-k3s"]
    
    Svc -- "Selector match" --> Pods
    Deploy -- "Selector match" --> Pods
    Deploy -. "Name reference" .-> CM
```

As illustrated above:
- **Service** and **Deployment** do **not** directly connect to each other. Instead, they both identify the same **Pods** independently using a **Selector** (`app: hello-k3s`).
- The **Deployment** finds and mounts the **ConfigMap** explicitly by its literal **Name** (`hello-k3s-app`).

### Step 3: Watch the pod start

```bash
kubectl get pods --watch
# Wait until STATUS = Running, then Ctrl-C
```

```bash
# If it gets stuck in Pending or Error:
kubectl describe pod -l app=hello-k3s
kubectl logs -l app=hello-k3s
```

### Step 4: Access the service

```bash
curl http://localhost:30500 | jq
# {"message": "Hello from K3s!", "pod": "hello-k3s-deployment-xxxx"}
```

From a remote machine (replace with your Jetson IP):

```bash
curl http://<jetson-ip>:30500
```

### Step 5: Explore

```bash
# Where is the code coming from?
kubectl describe configmap hello-k3s-app

# What does the pod see on disk?
kubectl exec -it deployment/hello-k3s-deployment -- ls /app
kubectl exec -it deployment/hello-k3s-deployment -- cat /app/app.py

# Scale to 2 replicas and watch different pod names appear in responses
kubectl scale deployment hello-k3s-deployment --replicas=2
for i in $(seq 6); do curl -s http://localhost:30500; echo; done

kubectl scale deployment hello-k3s-deployment --replicas=1
for i in $(seq 6); do curl -s http://localhost:30500; echo; done
```

### Step 6: Clean up

```bash
kubectl delete -f demo/01-hello-k3s/
```

---

## Demo 2: GPU Workload on Jetson

**`[Jetson]` only** — Review conceptually on Ubuntu workstation.

**Files:** `demo/02-gpu-workload/`

### Step 1: Enable GPU support

If you used the basic K3s install command (without the setup script), run:

```bash
bash demo/02-gpu-workload/setup-gpu-runtime.sh
```

Then deploy the NVIDIA Kubernetes device plugin — a DaemonSet that advertises GPU resources to the K3s scheduler:

```bash
kubectl apply -f demo/02-gpu-workload/nvidia-device-plugin.yaml
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset
```

### Step 2: Verify GPU capacity

```bash
# The node should now advertise GPUs as a schedulable resource
kubectl describe node | grep -A10 "Capacity:"
# Expected line: nvidia.com/gpu: 4
```

> **Why 4 and not 1?** Jetson has a single integrated GPU, but the assignment stack (`vllm` + `asr` + `tts`) needs three pods that each request `nvidia.com/gpu: 1`. The provided `nvidia-device-plugin.yaml` enables **time-slicing** with `replicas: 4`, telling the scheduler the GPU is "shareable up to 4 ways". Each pod still gets full GPU access at runtime — time-slicing is purely a scheduling primitive on Jetson where the iGPU already supports concurrent kernels. If you see `nvidia.com/gpu: 1`, the time-slicing ConfigMap was not applied — re-apply the plugin manifest.

### Step 3: Run a GPU test pod

```bash
kubectl apply -f demo/02-gpu-workload/gpu-test-pod.yaml
kubectl wait --for=condition=Succeeded pod/gpu-test --timeout=120s
kubectl logs gpu-test
```

If `nvidia-smi` output appears in the logs, GPU access is working end-to-end.

### Step 4: How is GPU access different from Docker Compose?

In short: instead of `runtime: nvidia` you declare `resources.limits: {nvidia.com/gpu: 1}` and the scheduler routes the pod to a node that has GPU capacity. See the full Compose-to-K3s mapping in the **Reference: Docker Compose → K3s Conversion** section at the bottom of this README.

```bash
kubectl delete pod gpu-test
```

---

## Part 4: Loading Images into K3s (Required for the Assignment)

> **Read this before applying the assignment YAMLs**, otherwise your pods will sit in `ImagePullBackOff` forever.
>
> **Prerequisite 1 — Lab 2 is done**: this part assumes Lab 2 is finished — i.e. you have rewritten the Lab 2 Dockerfile, written your own `api.py`, and successfully run `docker build -t faster-whisper:fastapi .`. If `docker images | grep faster-whisper` returns nothing on this machine, **finish Lab 2 first**; Lab 4's `asr` deployment cannot proceed without that image (or a public substitute, see §4.7 below).
>
> **Prerequisite 2 — model weights are pre-staged**: vLLM in this assignment loads the **Qwen3-4B-quantized.w4a16** weights from a hostPath volume mounted at `/opt/models/`. The TA team has pre-deployed the following directories to every Jetson host:
>
> ```
> /opt/models/Qwen3-4B-quantized.w4a16     ~3.3 GB  (used by vllm.yaml)
> /opt/models/Qwen3-ASR-0.6B               ~1.8 GB  (reference, not used by Lab 4)
> /opt/models/Qwen3-TTS-12Hz-0.6B-Base     ~2.4 GB  (reference, not used by Lab 4)
> ```
>
> Verify with `ls /opt/models/`. If `Qwen3-4B-quantized.w4a16` is missing or empty, **stop and contact a TA**; do not try to download it yourself (HuggingFace gating + Jetson network restrictions will burn hours). vLLM will not start without these weights.

### 4.1 Why this is necessary

Docker and K3s keep their images in **two completely separate stores**:

| Tool | Where images actually live | What lists them |
|---|---|---|
| `docker` | `/var/lib/docker/` (Docker's own image DB + overlay2) | `docker images` |
| K3s | depends on the K3s install path (both use namespace `k8s.io`):<br>• `/var/lib/containerd/` — Jetson, when K3s was installed with `--container-runtime-endpoint unix:///run/containerd/containerd.sock` (i.e. the path in §1.1 above)<br>• `/var/lib/rancher/k3s/agent/containerd/` — Workstation default, when K3s runs its embedded containerd | `sudo k3s crictl images` <br>or `sudo ctr -n k8s.io images ls` |

`docker pull` stores images in Docker's private DB; `kubectl apply` reads from the containerd `k8s.io` namespace. They never sync automatically — even when both happen to be on the same machine and even when both ultimately run via the same containerd daemon. So an image visible to `docker images` is **invisible** to K3s. You must move it across explicitly with `docker save` (export to a portable OCI tarball) and `ctr -n k8s.io images import` (re-ingest into K3s's namespace).

### 4.2 Assignment images and where they come from

The four assignment services use these images:

| Service | Image | Source | How to load |
|---|---|---|---|
| `vllm` | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` | Public (ghcr.io, ~9 GB) | `crictl pull` |
| `asr` | `faster-whisper:fastapi` | **Built locally in Lab 2** | `docker save` → `ctr import` |
| `tts` | `dustynv/kokoro-tts:fastapi-r36.4.0-cu128-24.04` | Public (Docker Hub, ~5 GB) | `crictl pull` |
| `open-webui` | `ghcr.io/open-webui/open-webui:main` | Public (ghcr.io, ~1 GB) | `crictl pull` |

### 4.3 Pulling the public images directly into K3s

K3s normally pulls public images on first use, but doing it ahead of time lets you see download progress and catch network errors:

```bash
sudo k3s crictl pull ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
sudo k3s crictl pull docker.io/dustynv/kokoro-tts:fastapi-r36.4.0-cu128-24.04
sudo k3s crictl pull ghcr.io/open-webui/open-webui:main
```

### 4.4 Loading the Lab 2 image (`faster-whisper:fastapi`) into K3s

This is the image you built in **Lab 2** — your own `Dockerfile` + `api.py` that exposes `/health`, `/v1/models`, and `/v1/audio/transcriptions` on port 5092. Lab 4 reuses this image as-is. Moving it into K3s is a 3-step process (Steps 1–3 below); but first, make sure the image actually exists.

#### Step 0: does the image already exist on this machine?

```bash
docker images | grep faster-whisper
# REPOSITORY        TAG       ...   SIZE
# faster-whisper    fastapi   ...   1–7 GB     <-- this is what you need
```

There are three cases:

**Case A — image is there.** Skip to Step 1; just reuse it. Lab 4 does not require you to rebuild every time.

**Case B — image is gone but your Lab 2 source is still on the machine.** Rebuild it from your Lab 2 folder. You need exactly two files inside that folder:

- `Dockerfile` — **the one you rewrote in Lab 2**, not the 460-byte placeholder shipped in `Lab2/faster-whisper/Dockerfile`. That placeholder is a fragment of the `jetson-containers` framework (`ARG BASE_IMAGE` with no default, references to `install.sh`/`build.sh` that depend on framework env vars). Running `docker build .` on it will fail with `BASE_IMAGE must not be empty`. Your own Dockerfile must:
  - choose a real base image (e.g. `python:3.9-slim` for a CPU build, or a `dustynv/...-r36.4.0` image for GPU acceleration),
  - install your runtime deps (`faster-whisper`, `fastapi` or `flask`, `uvicorn`, `ffmpeg`, …),
  - `COPY api.py` into the image,
  - `EXPOSE 5092` and set a `CMD` that starts your HTTP server.
- `api.py` — the FastAPI/Flask server you wrote, listening on `0.0.0.0:5092`, implementing `/health`, `/v1/models`, `/v1/audio/transcriptions`.

Then:

```bash
cd /path/to/your-lab2-submission           # the folder with your Dockerfile + api.py
ls Dockerfile api.py                       # both must be present

docker build -t faster-whisper:fastapi .   # tag exactly as faster-whisper:fastapi
```

Build time and image size depend on the base you picked:

| Base image used in your Lab 2 Dockerfile | Build time | Final image size | Inference |
|---|---|---|---|
| `python:3.9-slim` (or similar) | ~2–3 min | ~1.2 GB | CPU only |
| `dustynv/faster-whisper:r36.4.0-cu128-24.04` (or similar L4T base with CUDA pre-installed) | ~5–10 min | ~5–7 GB | GPU accelerated |
| `nvcr.io/nvidia/l4t-...` + manual install | 30+ min | 5–7 GB | GPU accelerated |

> Note: this **rebuilds** your image; it does **not** redo your Lab 2 work. If the rebuild fails, the problem is in your `Dockerfile` / `api.py`, not in Lab 4 — go back and finish Lab 2 first.

**Case C — both the image and the source are gone.** Recover your Lab 2 work (re-clone your private repo, or re-download what you submitted to Blackboard), put it back somewhere on this machine, then follow Case B.

#### Step 1: save the Docker image to a portable tarball

```bash
docker save faster-whisper:fastapi -o /tmp/fw.tar
```

Time and tarball size are roughly the same as the image size from Step 0 (~30 s for a 1 GB slim image, ~3 min for a 7 GB CUDA-accelerated image).

#### Step 2: import the tarball into K3s's containerd (namespace `k8s.io`)

```bash
sudo ctr -n k8s.io images import /tmp/fw.tar         # unpack ≈ same as save time
sudo rm /tmp/fw.tar                                  # tarball is root-owned
```

#### Step 3: verify and re-deploy

```bash
sudo k3s crictl images | grep faster-whisper
sudo kubectl rollout restart deployment asr
sudo kubectl get pod -l app=asr -w
```

`STATUS=Running` and event `Container image "faster-whisper:fastapi" already present on machine` means it worked.

> **Disk space**: Steps 1–2 briefly hold both the tarball **and** the unpacked copy in K3s's image store at the same time, so plan for **roughly 2× your image size in free space on `/`** (e.g. ~3 GB free for a 1 GB image, ~15 GB free for a 7 GB image). If your disk is over ~85% full, the kubelet may evict idle pods mid-import. Free space first with `docker image prune -a` to remove any old Lab 1/2/3 images you no longer need.

### 4.5 Common error: `ImagePullBackOff` after `kubectl apply`

```bash
$ kubectl get pod
NAME                  READY   STATUS             RESTARTS   AGE
asr-xxxx-yyyy         0/1     ImagePullBackOff   0          2m
```

```bash
$ kubectl describe pod asr-xxxx-yyyy
...
Events:
  Warning  Failed  ...  Failed to pull image "faster-whisper:fastapi": ...
                       pull access denied, repository does not exist or may
                       require authorization
```

This message is misleading. It does **not** mean a permissions issue — it means K3s tried to pull `faster-whisper:fastapi` from `docker.io` and didn't find it. The fix is §4.4 above (after `ctr import`, the rollout restart there picks the image up).

Tip: setting `imagePullPolicy: IfNotPresent` in your YAML also helps — it tells K3s to skip the registry pull if the image is already in containerd.

### 4.6 Quick sanity check before applying the assignment

Two things must be true on this machine before `kubectl apply`:

```bash
# 1. all four images are loaded into K3s containerd
sudo k3s crictl images | grep -E 'vllm|kokoro|faster-whisper|open-webui'
# expected: 4 lines (vllm, kokoro-tts, faster-whisper, open-webui)

# 2. vLLM model weights are present on the host (mounted via hostPath)
ls /opt/models/Qwen3-4B-quantized.w4a16
# expected: a non-empty directory containing config.json, *.safetensors, etc.
```

If either check fails, fix it before `kubectl apply` — apply will succeed but the pods will hang/crash and you will spend hours diagnosing.

### 4.7 Image selection caveats (common mis-picks for `asr`)

If you don't want to use your own Lab 2 build (for instance, the build keeps failing and the deadline is close), you may be tempted to point the `asr` deployment at a public faster-whisper image. Two specific images cause confusing failures on Jetson:

**❌ `lewangdev/faster-whisper:latest`** — this is an x86_64-only image. K3s will pull it successfully, then the container immediately exits with:

```
exec /opt/nvidia/nvidia_entrypoint.sh: exec format error
```

This is a CPU-architecture mismatch (amd64 binary on arm64 hardware), not a config issue. There is no Jetson-friendly arm64 variant of this tag. Pick a different image.

**⚠️ `dustynv/faster-whisper:r36.4.0-cu128-24.04`** — this image is arm64 / Jetson-native (good), but its default `ENTRYPOINT` runs the upstream test scripts and then exits 0. Used as a server it will `Completed` -> restart -> `Completed` -> ... in a CrashLoopBackOff with hundreds of restarts and **exit code 0**. To use it as the `asr` server you must override the entrypoint and supply your own server, e.g. by mounting your `api.py` from a hostPath and adding to the deployment:

```yaml
command:
- sh
- -c
- pip3 install fastapi uvicorn python-multipart && python3 /workspace/api.py
volumeMounts:
- {name: asr-api, mountPath: /workspace/api.py, subPath: api.py}
volumes:
- name: asr-api
  hostPath:
    path: /home/<your-user>/path/to/Lab2/faster-whisper
```

Either way, the cleanest path is the standard one in §4.4: build `faster-whisper:fastapi` from your Lab 2 source, `docker save` + `ctr import`, and reference `image: faster-whisper:fastapi` in `asr.yaml`.

### 4.8 Common error: `vllm` pod in `CrashLoopBackOff`

`ImagePullBackOff` (§4.5) means K3s couldn't get the image. **`CrashLoopBackOff`** is different: the image was found, the container started, but the process inside died. For `vllm` on Jetson Orin NX (16 GB shared RAM/VRAM) the cause is almost always one of:

```bash
sudo kubectl get pod -l app=vllm
# NAME             READY   STATUS             RESTARTS   AGE
# vllm-xxxx-yyyy   0/1     CrashLoopBackOff   8 (1m ago) 25m

# Read the logs from the last failed attempt — the live logs are usually empty.
sudo kubectl logs -l app=vllm --previous --tail=80
```

Look for these markers in the previous-attempt logs:

| What you see | What it means | Fix |
|---|---|---|
| `torch.cuda.OutOfMemoryError` <br>or `Killed` (the pod just disappears) <br>or `OOMKilled` in `kubectl describe pod` | The 4B quantized model + KV cache wouldn't fit alongside `asr` and `tts`. | Lower memory pressure: `--gpu-memory-utilization 0.40`, `--max-model-len 2048`, `--max-num-batched-tokens 1024`. Apply only one change at a time and re-deploy. |
| `FileNotFoundError: ... config.json` <br>or `OSError: ... Qwen3-4B-quantized.w4a16` | The hostPath volume is empty / wrong path / model directory missing. | `ls /opt/models/Qwen3-4B-quantized.w4a16` must list `config.json`. If empty, ask a TA — see Prerequisite 2 at the top of Part 4. |
| `address already in use: 0.0.0.0:8000` | Another process on this Jetson is already bound to port 8000 (e.g. an old `docker run` of vLLM, or a shared user's vLLM). `hostNetwork: true` means there is no port isolation. | `sudo ss -tlnp | grep 8000` to find the offender, kill it, then `kubectl rollout restart deployment vllm`. |
| `RuntimeError: CUDA error: ...` <br>or hangs at "Loading model" for >5 min | First-load cuBLAS/cuDNN compilation is slow on Jetson — usually fine after one full restart. If it persists, try `kubectl delete pod -l app=vllm` once. | Wait. Allow up to 90 s for first model load. If it still hangs, restart the pod. |

> **OOM is not just a vLLM problem.** All three GPU pods (`vllm` + `asr` + `tts`) share the same 16 GB iGPU pool. If you reduced `--gpu-memory-utilization` and vLLM still OOMs, check that `asr` and `tts` are actually idle — running speech inference on all three simultaneously can push the total over the limit.

---

## Reference: Docker Compose → K3s Conversion

| Docker Compose concept | K3s / Kubernetes equivalent |
|---|---|
| `services:` | `Deployment` + `Service` (one pair per service) |
| `image:` | `spec.containers[].image` |
| `ports: "3000:8080"` | `Service` with `type: NodePort` |
| `environment:` | `env:` in container spec, or `ConfigMap` |
| `volumes: "host:container"` | `hostPath` volume |
| Named volume (`volumes: data:`) | `PersistentVolumeClaim` |
| Service DNS names | `ClusterIP` Service + K8s DNS |
| `runtime: nvidia` | `resources.limits: {nvidia.com/gpu: 1}` |
| `network_mode: host` | `hostNetwork: true` in pod spec |
| `shm_size: "8g"` | `emptyDir: {medium: Memory, sizeLimit: 8Gi}` |
| `depends_on:` | `readinessProbe` on the dependency |
| `restart: always` | Deployment `restartPolicy: Always` (default) |

### Complete example: TTS service

**Docker Compose (Lab 3):**

```yaml
tts:
  image: dustynv/kokoro-tts:fastapi-r36.4.0-cu128-24.04
  runtime: nvidia
```

**K3s equivalent:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tts
  template:
    metadata:
      labels:
        app: tts
    spec:
      containers:
      - name: tts
        image: dustynv/kokoro-tts:fastapi-r36.4.0-cu128-24.04
        ports:
        - containerPort: 8880
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: tts
spec:
  type: ClusterIP
  selector:
    app: tts
  ports:
  - port: 8880
    targetPort: 8880
```

> **Note on `nvidia.com/gpu: 1` and time-slicing:** the line `limits: {nvidia.com/gpu: 1}` requests **one GPU slot** for this pod. With time-slicing enabled by `nvidia-device-plugin.yaml` (see Demo 2 / §Step 2), the Jetson's single iGPU advertises `nvidia.com/gpu: 4` of capacity, so up to four pods that each ask for `1` can run concurrently. They share the GPU at runtime — Jetson's iGPU supports concurrent kernels — and each pod still sees the full device.

> **Note on `network_mode: host` and vLLM:** vLLM in Lab 3 used `network_mode: host` for performance. In K3s the equivalent is `hostNetwork: true` in the pod spec. However, when `hostNetwork: true` is set, other pods must reach vLLM via the node IP (e.g., `http://192.168.1.x:8000`) rather than via a ClusterIP Service DNS name. The assignment asks you to handle this trade-off explicitly.
