# Deployment Guide

How `imagedetection-api` (the FastAPI wrapper in [main.py](main.py) around the
change-detection pipeline) gets from a commit to a running pod, and the
Kubernetes concepts behind the manifests in this repo (`k8s-deploy.yml`,
`k8s-secrets.yml`, `k8s-services.yml`, `k8s-ingress.yml`).

## File map

| File | Role |
|---|---|
| [main.py](main.py) | FastAPI app: HTTP endpoints, submits jobs to Celery |
| [pipeline.py](pipeline.py) | Pure pipeline logic (no FastAPI/Celery imports) — the actual SSIM/DINOv2/report work |
| [celery_app.py](celery_app.py) | Celery app: Redis broker/result-backend config, eager-mode fallback for local dev |
| [tasks.py](tasks.py) | The one Celery task (`run_analysis_task`), wraps `pipeline.run_analysis` |
| [tracer.py](tracer.py) | Datadog APM (`ddtrace`) + custom dogstatsd metrics (`datadog`), imported first by both `main.py` and `tasks.py` |
| [entrypoint.sh](entrypoint.sh) | Docker `ENTRYPOINT`; dispatches to uvicorn / celery worker / celery beat / flower by role argument |
| [Dockerfile](Dockerfile) | Single image used for every role above |
| `k8s-*.yml` | Kubernetes manifests deploying all of the above |

## 1. Architecture at a glance

```
 developer push
       |
       v
 Azure DevOps (ADO) Pipeline
   ├─ Build stage:  docker build -> tag -> push        --> Azure Container Registry (ACR)
   ├─ Fetch stage:  AzureKeyVault@2 task pulls secrets  --> ADO pipeline variables (masked)
   └─ Deploy stage: kubectl/helm apply manifests        --> AKS cluster (Rancher-managed)
                          |
                          v
                 +-------------------------+
                 |     AKS  (Rancher)      |
                 |    imagedetection       |  namespace
                 |    --------------       |
                 |  Deployment pods:       |
                 |   [api][worker]         |<--- one pod, two containers (sidecar)
                 |        |                |
                 |  Deployment (1 replica):|
                 |   [flower]              |<--- Celery monitoring UI
                 +--------|----------------+
                          v
                 Azure Cache for Redis   (Celery broker + result backend)
                          |
                          v
                    Datadog Agent (DaemonSet)  <--- traces/metrics from every role via tracer.py
```

- **ADO** builds the Docker image and pushes it to **ACR**; the same pipeline
  then deploys the manifests in this repo to **AKS**.
- **AKS** is provisioned/managed day-to-day through **Rancher** (RBAC, cluster
  lifecycle, monitoring rollups), while ADO talks to it as an ordinary
  Kubernetes API endpoint.
- Each `imagedetection-api` pod runs **two containers**: `api` (serves HTTP,
  submits Celery tasks) and `worker` (a sidecar running a Celery worker
  process against **Azure Cache for Redis**, doing the actual SSIM/DINOv2
  inference). A separate single-replica `imagedetection-flower` Deployment
  runs the Celery monitoring UI. All three, plus local `uvicorn --reload`,
  run from the **same image**, dispatched by [entrypoint.sh](entrypoint.sh)'s
  first argument (`api` / `worker` / `flower` / `beat`).

## 2. CI/CD: Docker build → ACR → AKS via ADO

### 2.1 Pipeline stages

1. **Build**
   ```yaml
   - task: Docker@2
     inputs:
       command: buildAndPush
       repository: imagedetection-api
       containerRegistry: acr-imagedetection-connection   # ADO service connection to ACR
       tags: |
         $(Build.BuildId)
         latest
   ```
   The image is built once from the [Dockerfile](Dockerfile) and tagged with
   the immutable ADO `Build.BuildId` — that tag is what actually gets deployed
   (never `latest`, so a rollback is just re-deploying a previous `BuildId`).

2. **Fetch secrets from Key Vault**
   ```yaml
   - task: AzureKeyVault@2
     inputs:
       azureSubscription: azure-imagedetection-connection
       KeyVaultName: kv-imagedetection-prod
       SecretsFilter: 'redis-connection-string,datadog-api-key'
       RunAsPreJob: true
   ```
   This pulls secret values into ADO pipeline variables for the duration of
   the run. ADO automatically **masks** them in logs. They're used two ways
   depending on the target cluster's capabilities:
   - Preferred: the AKS cluster has the **Secrets Store CSI Driver** with the
     Azure Key Vault provider installed, so pods pull secrets directly from
     Key Vault at mount time via workload identity — ADO's pipeline variables
     are never actually needed at deploy time, only `SecretProviderClass`
     objects (`k8s-secrets.yml`) which reference the Key Vault by name, not
     by value.
   - Fallback: a **Replace Tokens** (or `envsubst`) pipeline task substitutes
     `#{REDIS_CONNECTION_STRING}#`-style placeholders in `k8s-secrets.yml`
     with the fetched ADO variables immediately before `kubectl apply`, so
     the plaintext only exists transiently on the build agent.

   Both paths are shown in [k8s-secrets.yml](k8s-secrets.yml); pick one.

3. **Deploy**
   ```yaml
   - task: Kubernetes@1
     inputs:
       connectionType: Kubernetes Service Connection
       kubernetesServiceEndpoint: aks-imagedetection-rancher   # kubeconfig obtained via Rancher
       namespace: imagedetection
       command: apply
       arguments: -f k8s-secrets.yml -f k8s-deploy.yml -f k8s-services.yml -f k8s-ingress.yml
   - script: |
       kubectl set image deployment/imagedetection-api \
         api=$(ACR_LOGIN_SERVER)/imagedetection-api:$(Build.BuildId) \
         worker=$(ACR_LOGIN_SERVER)/imagedetection-api:$(Build.BuildId) \
         -n imagedetection
   ```

### 2.2 How the artifact actually reaches AKS

ACR and AKS are linked with `az aks update --attach-acr`, which grants the AKS
kubelet identity `AcrPull` on the registry — nodes pull images directly, no
`imagePullSecret` needed. ADO's role is only to (a) push the tagged image to
ACR and (b) tell the cluster, via `kubectl set image` / `kubectl apply`, which
tag to run. The **Kubernetes service connection** in ADO holds a kubeconfig
for the cluster; because the cluster is **imported into and managed by
Rancher**, that kubeconfig is generated from Rancher's downstream-cluster API
(`Cluster > Kubeconfig File` or a dedicated Rancher service account token)
rather than directly from `az aks get-credentials` — this way, access is still
subject to Rancher's own RBAC/project bindings, and Rancher's UI continues to
show accurate rollout status for anything ADO deploys.

## 3. Sidecar container architecture (Celery + Azure Redis as broker)

Each pod is **one Deployment, two containers** (`k8s-deploy.yml`), both using
the same image and [entrypoint.sh](entrypoint.sh), just a different first
argument:

| Container | Command | Role |
|---|---|---|
| `api` | `entrypoint.sh api` → `uvicorn main:app` | Serves HTTP. `POST /analyze` calls `run_analysis_task.delay(...)` (Celery), which enqueues a task onto Redis and returns `202 Accepted` with the Celery task id as `job_id`. |
| `worker` | `entrypoint.sh worker` → `celery -A celery_app worker` | Consumes tasks from the same Redis broker, runs `pipeline.run_analysis` (SSIM/DINOv2/footprint scoring), writes the result back to Redis (the result backend) and metrics/figures to the shared PVC. |

A third, independent single-replica Deployment, `imagedetection-flower`, runs
`entrypoint.sh flower` → `celery -A celery_app flower`, Celery's web-based
monitoring UI (task history, retries, per-worker throughput) — see
`k8s-deploy.yml`. It is **not** a container in the `api`/`worker` pod: it has
no request-serving role, autoscaling it would be meaningless (there's exactly
one thing to look at), and its only meaningful failure mode (a dashboard
being briefly down) is unrelated to `api`/`worker` availability.

`entrypoint.sh` is the single place all four roles (`api`, `worker`, `beat`,
`flower`) start from. Before starting anything it:
1. Runs `check_tracer`: `python3 -c "import tracer"`, so a broken Datadog
   config ([tracer.py](tracer.py) failing to import — e.g. a bad
   `DD_AGENT_HOST`, or `ddtrace`/`datadog` missing from the image) fails the
   container immediately at startup, not silently on the first request.
2. Runs `wait_for_redis`: polls `REDIS_URL` with `redis.ping()` for up to 30s
   before handing off to uvicorn/celery, so a container doesn't start serving
   traffic (or claim to be a healthy worker) against a broker that isn't
   reachable yet — relevant right after a fresh pod start, before Azure Cache
   for Redis DNS/network has settled.
3. `exec`s the actual process (`uvicorn`, `celery ... worker`, `celery ...
   beat`, or `celery ... flower`) as PID 1, so it receives `SIGTERM` directly
   during a rolling update or pod eviction instead of it going to a shell
   wrapper that doesn't forward it.

**Azure Cache for Redis** is Celery's broker *and* result backend:
- It's an external managed service (not something running inside the
  cluster), reached via `REDIS_URL` injected from the `imagedetection-secrets`
  Secret (see [celery_app.py](celery_app.py)).
- `api` only ever calls `.delay()` (enqueue) and `AsyncResult(...)` (read
  status/result); `worker` only ever pulls the next task and writes its
  result. Neither container needs the other's code path, so either can be
  redeployed, crash, or be OOMKilled independently without losing queued
  jobs — Celery's own delivery/acknowledgment semantics (`acks_late=True` on
  the task) handle that, not application code.
- If `REDIS_URL` is unset (e.g. local dev with no Redis running),
  `celery_app.py` sets `task_always_eager=True`: `.delay()` runs the task
  synchronously in the calling process instead of going through a broker at
  all, so the API is fully usable standalone with zero extra infrastructure.
  `task_eager_propagates=False` and `task_store_eager_result=True` keep this
  path's behavior consistent with the real async path — a bad request still
  comes back as a normal `FAILURE` `AsyncResult`, not an unhandled exception,
  and `GET /analyze/{job_id}` can still look the result up by id afterwards.

## 3.1 Observability wiring: tracer.py

[tracer.py](tracer.py) is imported as the *first* import in both
[main.py](main.py) and [tasks.py](tasks.py) — before FastAPI, Celery, or
redis-py are imported — because `ddtrace.patch_all()` monkey-patches those
libraries to emit spans automatically, and that patching has to happen before
the libraries are otherwise used. It:
- Calls `ddtrace.patch_all()` and lets `ddtrace` itself read `DD_SERVICE`,
  `DD_ENV`, `DD_VERSION`, `DD_AGENT_HOST`, `DD_LOGS_INJECTION` straight from
  the environment (all set in `k8s-deploy.yml`), so APM traces and log
  correlation work with no code beyond the import.
- Initializes the `datadog` package's dogstatsd client against the same
  `DD_AGENT_HOST`, exposing `tracer.incr(...)` / `tracer.timing(...)` for
  custom business metrics — `main.py` increments `imagedetection.analyze.queued`
  on every submitted job, `tasks.py` increments
  `imagedetection.analyze.succeeded` / `.failed` and times
  `imagedetection.analyze.duration_ms` around the actual pipeline run.
- Is a no-op everywhere (tracer disabled, `incr`/`timing` return immediately)
  when `DD_AGENT_HOST` isn't set, so local dev never needs a Datadog Agent
  running to import `main.py`/`tasks.py` successfully.

## 4. Kubernetes fundamentals (as applied in this repo)

### Pods, Nodes, Containers
- A **container** is one running process image (here: the same
  `imagedetection-api` image, run with two different commands).
- A **Pod** is the smallest deployable unit — one or more containers that
  always land on the same **Node** and share network/IP and, if declared,
  storage volumes. This repo's Pod = `api` + `worker` containers.
- A **Node** is a VM (an AKS VMSS instance) that runs the `kubelet` and hosts
  some number of Pods, scheduled by the control plane based on each Pod's
  requested `resources`.
- A **Deployment** (`k8s-deploy.yml`) is a controller that keeps a desired
  number of identical Pod replicas running, handles rolling updates (new
  `BuildId` image → old Pods drained, new Pods started, one/few at a time),
  and self-heals (a crashed Pod is replaced automatically).

### Services and Ingress
- A **Service** (`k8s-services.yml`) is a stable virtual IP + DNS name
  (`imagedetection-api.imagedetection.svc.cluster.local`) load-balancing
  traffic across whichever Pods currently match its label selector — Pods
  come and go (rescheduled, scaled, rolled) but the Service address doesn't.
- An **Ingress** (`k8s-ingress.yml`) is an HTTP(S) entrypoint from outside the
  cluster: the NGINX ingress controller terminates TLS (certificate issued by
  cert-manager against `letsencrypt-prod`) and routes `imagedetection.example.com`
  to the `imagedetection-api` Service.

### HorizontalPodAutoscaler (HPA)
Defined at the bottom of `k8s-deploy.yml`: the HPA controller polls the
metrics-server for each Pod's CPU/memory utilization every ~15s and adjusts
`replicas` on the Deployment between `minReplicas: 2` and `maxReplicas: 8` to
hold utilization near the configured target (70% CPU / 80% memory). A 300s
`stabilizationWindowSeconds` on scale-down avoids flapping (rapidly scaling
down then back up under bursty load); scale-up has no window so it reacts
immediately, adding up to 2 Pods per minute. Because `worker` and `api` are
containers in the *same* Pod, this HPA scales both together — a
worker-specific signal (e.g. Redis queue depth via KEDA) would be the next
step if the two ever need to scale independently.

### Volume mounting & read/write access
- `imagedetection-outputs` (`k8s-deploy.yml`) is a **PersistentVolumeClaim**
  backed by Azure Files, requested with `ReadWriteMany` — the pipeline's
  `analysis_output/` directory (metrics.json, per-pair PNGs, the built PDF
  report) needs to be **written by both containers** in a Pod and **read
  consistently across all replicas**, which rules out a `ReadWriteOnce` Azure
  Disk (only attachable to one node at a time).
- Both `api` and `worker` mount it at the same path, `/app/analysis_output`,
  matching `OUT_DIR` in `main.py`/`compare_images.py` — so a report the
  worker builds is immediately visible to whichever `api` replica serves the
  next `GET /report`.
- The Pod's `securityContext` sets `runAsNonRoot: true`, `runAsUser: 10001`
  (matching the `appuser` created in the [Dockerfile](Dockerfile)), and
  `fsGroup: 10001` so the mounted volume is group-writable by that same
  non-root UID — containers never run as root, and the only writable
  filesystem path is the explicitly mounted volume (the rest of the
  container filesystem is the read-only image layer).
- The Key Vault CSI volume (`akv-secrets`) is mounted `readOnly: true` — the
  app only ever needs to read secret files/env values, never write them.

## 5. Observability: Datadog

- The cluster runs the **Datadog Agent as a DaemonSet** (one Agent per Node,
  installed cluster-wide, outside this repo's manifests) which:
  - Collects **infrastructure metrics** (CPU/memory/network per Pod/Node) —
    this is what the HPA's metrics-server also draws on, and what backs the
    Kubernetes Pod/Deployment dashboards in Datadog.
  - Tails **container logs** automatically via Kubernetes log
    autodiscovery — no log-shipping sidecar needed.
  - Receives **APM traces and dogstatsd metrics** from the
    `api`/`worker`/`flower` containers, via [tracer.py](tracer.py) (see §3.1)
    over `DD_AGENT_HOST` (injected from the pod's `status.hostIP`, so each
    container reports to the Agent running on its own Node).
- `k8s-deploy.yml` sets `DD_SERVICE`, `DD_ENV`, `DD_VERSION` (the latter set
  to the same `Build.BuildId` used for the image tag) per container — `api`
  reports as `imagedetection-api`, `worker` as `imagedetection-worker` — plus
  `DD_LOGS_INJECTION=true`, so that:
  - Every trace/log is automatically correlated by `trace_id`, letting you
    jump from "this request was slow" straight to its logs.
  - Datadog's Deployment view can distinguish "errors introduced by
    `BuildId` 4821" from prior versions during a rollout.
- Pod labels `tags.datadoghq.com/service|env|version` let Datadog tag
  infrastructure metrics with the same service/env/version dimensions as the
  APM traces, so each service's page in Datadog covers requests, logs, and
  the Pods/Nodes they ran on.
- `main.py`'s own `GET /metrics` endpoint returns the *pipeline's* JSON
  results (SSIM/DINOv2 scores) — it is an application endpoint, not a
  Prometheus/OpenMetrics scrape target, so it is intentionally not wired into
  Datadog's OpenMetrics check. Business metrics (jobs queued/succeeded/failed,
  job duration) instead go through `tracer.incr()`/`tracer.timing()`'s
  dogstatsd client, described in §3.1.
- `entrypoint.sh`'s `check_tracer` step means a container that can't reach
  Datadog correctly (or is missing `ddtrace`/`datadog` from the image) fails
  its `CrashLoopBackOff` immediately and visibly, rather than running with
  silently-broken observability.

## 6. Local development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# REDIS_URL and DD_AGENT_HOST both unset:
#  - Celery runs tasks eagerly, in-process (celery_app.py) — no broker needed.
#  - tracer.py's Datadog tracing/metrics are no-ops — no Agent needed.
```

To exercise the real Celery worker + Redis broker path locally (e.g. before
trusting a change to `pipeline.py`/`tasks.py` in the actual pod):

```bash
redis-server &                                          # or: docker run -p 6379:6379 redis:7
REDIS_URL=redis://localhost:6379/0 uvicorn main:app --reload           # terminal 1
REDIS_URL=redis://localhost:6379/0 ./entrypoint.sh worker               # terminal 2
REDIS_URL=redis://localhost:6379/0 ./entrypoint.sh flower               # terminal 3 (optional), http://localhost:5555
```

```bash
docker build -t imagedetection-api:local .
docker run -p 8000:8000 -e REDIS_URL=redis://host.docker.internal:6379 imagedetection-api:local            # api (default CMD)
docker run -e REDIS_URL=redis://host.docker.internal:6379 imagedetection-api:local worker                  # worker
docker run -p 5555:5555 -e REDIS_URL=redis://host.docker.internal:6379 imagedetection-api:local flower      # flower
```
