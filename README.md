# Theft Detection System

Real-time edge video analytics for retail shoplifting detection. Frames flow from RTSP cameras
through ingest, ONNX/YOLOv8 inference, ByteTrack tracking and an EWMA suspicion engine, ending in
alerts and a live browser dashboard — all on one box, with no cloud hop.

The C++ services (ingest, inference) are the hot data path. The Python services (tracking, event,
display, control) handle tracking, alerting, the UI and fleet lifecycle.

---

## Performance

<!-- Measured with `tests/benchmark_pipeline.py` (see [Benchmarking](#benchmarking)) on 2026-10-28/29. -->

**Setup:**
- 8 simulated cameras at 15 fps each (120 fps in total): looped H.264 files (768×432 and 360×640)
  served by MediaMTX.
- Shared-memory transport and default settings, with 4 inference workers. YOLOv8 runs in FP16 through
  ONNX Runtime's TensorRT provider.
- The display service was running, with no viewer connected.

Latency covers each frame from the ingest RTSP read until tracking and alerting are done with it.

| | Laptop | Jetson Orin NX 16 GB |
|---|---|---|
| Hardware | i7-13700HX (24 threads), RTX 4070 Laptop 8 GB | Orin NX Super devkit, `MAXN_SUPER`, clocks not locked |
| Software | TensorRT 10.8, ONNX Runtime 1.22, CUDA 12.8 | JetPack 6.2 (L4T R36.4.3), TensorRT 10.7, ONNX Runtime 1.24, CUDA 12.6 |
| Measured window | 10 min | 3 min |
| Frames tracked | 119.9 fps | 119.8 fps |
| Frames dropped | 0.03% | 0.14% (age-based safety net) |

**End-to-end latency, ms**

| | mean | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| Laptop | 11.4 | 11.1 | 16.0 | 21.2 | 30.4 | 63.0 |
| Jetson | 41.2 | 41.1 | 60.3 | 71.8 | 76.7 | 82.4 |

**Where the time goes (mean, ms)**

| Hop | Laptop | Jetson |
|---|---|---|
| ingest → inference (shm + gRPC) | 0.8 | 1.7 |
| inference: queue wait + `infer()` + hop to tracking | 10.0 | 37.5 |
| &nbsp;&nbsp;of which `infer()` | 8.2 | 26.9 |
| tracking + alerting | 0.7 | 2.0 |

**CPU and memory per service.** CPU is in % of one core. RAM is PSS, which divides shared pages (the
shm ring, libraries) between the processes that map them, so the column adds up.

| Service | Laptop CPU | Laptop RAM | Jetson CPU | Jetson RAM |
|---|---|---|---|---|
| ingest | 122% | 637 MiB | 126% | 445 MiB |
| inference | 190% | 4,222 MiB | 220% | 1,637 MiB |
| tracking | 13% | 81 MiB | 34% | 81 MiB |
| display | 0.2% | 63 MiB | 0.5% | 57 MiB |
| GPU memory (inference) | | 680 MiB of VRAM | | 527 MiB, in shared RAM |
| **Total** | **325% of 2,400%** | **5,002 + 680 MiB** | **380% of 800%** | **2,220 + 527 MiB** |

On the Jetson the CPU and GPU share one 16 GB pool, so 2,220 + 527 MiB is the pipeline's whole
footprint. The 186 MiB of GPU buffers mapped into inference count only in the GPU row.

**Jetson power and thermals** (INA3221 rails, via `tegrastats`)

| | Load | Idle |
|---|---|---|
| Board input (`VDD_IN`) | 22.9 W | 5.8 W |
| CPU + GPU + CV rail | 12.2 W | 0.5 W |
| SoC rail | 4.7 W | 1.7 W |
| Energy | 191 mJ per frame, 5.2 fps per W | |
| Junction temperature (max) | 72.8 °C, still rising slowly after 3 min | 52.7 °C |
| GPU load (GR3D) | 69% average | |

Reading these numbers:

- **On the Jetson, inference sets the latency.** `infer()` takes 27 of the 41 ms, and most of the rest
  is queue wait: the 4 workers are close to saturated (queue depth 2–5).
- **Most of inference's RAM is allocated while the TensorRT model loads**, not while frames run. That
  is 3.1 GB of heap before the first frame on x86 and 1.1 GB on the Jetson.
<!-- - **Laptop latency moved by about 3 ms between sessions** (means of 8–11 ms), so only compare runs
  taken back to back. -->
- **The measurement doesn't disturb the numbers.** In an A/B test, alternating windows with the
  benchmark attached and detached, latency differed by no more than the windows differ among
  themselves.

The benchmark results are produced with a script which will be added shortly.
---

## Architecture

![System architecture](System.png)

The system has a micro-service architecture — see the `Services` [table](#services) below.

### Working/High Level Data Flow

**Ingest** pulls the RTSP cameras listed in `config.yaml` and publishes each frame into a shared-memory
ring. **Inference** runs YOLOv8 over those frames and pushes detections forward. **Tracking** turns
detections into track IDs, scores each track for suspicion, and raises alerts. **Display** reads frames
and detections back out of shared memory and serves the annotated video wall. **Control** starts and stops
the fleet.

Two properties shape everything else:

- **Pixels move once.** Frames go into the ring; the gRPC hops between services carry only metadata.
- **The flow is one-directional.** Nothing returns to ingest — each stage pushes forward and forgets, so a
  slow or dead consumer drops frames instead of stalling camera capture.

### Detailed Data Flow

**Ingest** spawns one `RtspReader` thread per stream and downscales every frame to fit 640×640 before 
publishing it through a `FrameWriter` — the same already-resized pixels on
either transport, so inference sees identical coordinates. All readers share **one** gRPC channel to
`INFERENCE_ADDR`, but each opens its **own** client-streaming `grpcStreamFrames` RPC. Under `shm` the
pixels go to the ring and the RPC carries metadata only; under `grpc` they ride inline in the same message.
Ingest also runs an **`IngestAdmin`** gRPC server, so cameras can be added, started, stopped and removed on
a *running* ingest with no restart.

**Inference** is a server, a worker pool and a detector. The server accepts the frame RPCs and pushes each
onto one shared `FrameQueue`. Worker threads pop from it, fetch the
pixels (from the ring or from the message), run the shared ONNX/YOLOv8 detector, and forward detections
plus frame metadata to tracking over a client-streaming RPC that **self-heals** — it reopens with capped
backoff if tracking restarts, losing only the frames sent while it was unreachable. The detector selects
its execution provider at load time: TensorRT → CUDA → CPU.

**Tracking** assigns track IDs, then runs each track through an EWMA suspicion score with hysteresis and a
sustain window to produce alerts. It keeps one tracker per `stream_id` behind an algorithm-agnostic
`Tracker` interface whose input carries pixels optionally, so image-based and motion-only algorithms both
drop in. The default is **ByteTrack**, which is IoU/Kalman only and ignores pixels — which is why
`TRACKING_PIXELS` defaults to `0`.

**Alerting** is a *library*, not a service: tracking imports `AlertEngine` from `event_service` and runs it
in-process, so a detection becomes an alert with no extra network hop. Alerts land in JSON and SQLite under
`EVENT_DATA_DIR`.

**Control** never forks or execs anything. It renders each service's launch-time config into an env file
and delegates the lifecycle to the process manager (`systemctl --user`), which leaves restart semantics and
log capture owned by systemd. It spans two planes: **lifecycle** (start/stop/restart + launch-time config,
via a `LifecycleBackend`) and **runtime** (camera add/start/stop on a live ingest, via `IngestAdminClient`; not added on display-service yet).
`SystemdBackend` and `ComposeBackend`(implementation in progress) both sit behind that one interface.

### Design Choices

**Frame transport** — a frame is copied into `/dev/shm` once and every consumer reads it in place by
`(stream_id, frame_id)`. The inline-gRPC path is **partially implemented** and needs further work.

**Torn-read safety** — each ring slot carries a seqlock, so a reader can never observe a half-written
frame.

**Transport handshake** — every service must agree on `shm` vs `grpc`(partially implemented, system will break on grpc). 
The first peer to start publishes its
choice into the segment; a peer wanting the other aborts with a diagnostic rather than reading garbage. The
mode *persists* in the segment, so switching means `rm /dev/shm/sec-sys-shm` and restarting every peer.

**Version handshake** — the segment header carries a magic number and a layout version, checked on the same
path. Any change to the shm structs bumps `SHM_VERSION`, so a stale segment or a mismatched build fails
fast instead of silently misreading. The layout is mirrored in Python for the display reader.

**Data plane in C++** — ingest and inference, where the per-frame work is.

**Everything above the hot path in Python** — tracking and alerting (per-frame, but never per-pixel), plus
the actual control plane: control and display. The display service owns no lifecycle logic of its own; its
buttons are a thin facade over `control_service.FleetController`.

**Back pressure** — deliberately absent. Both buffers drop rather than block: the ring laps, and the frame
queue evicts its oldest entry. A slow consumer loses frames instead of adding latency, which is the
right trade for a live view.

**Observability** — every service exposes Prometheus metrics on its own listener: per-hop latency,
end-to-end frame age, frames dropped, queue depth, per-stream FPS and liveness. Labels stay bounded —
`stream` and `class_id` only, never `track_id` or `frame_id`.


### Services

| Service | Language | Role |
|---|---|---|
| `ingest_service` | C++ | RTSP capture, shm ring writer, gRPC client, camera admin RPC |
| `inference_service` | C++ | gRPC server, YOLOv8 via ONNX Runtime (TensorRT/CUDA/CPU), worker pool |
| `tracking_service` | Python | gRPC server, ByteTrack association, suspicion + alerting |
| `event_service` | Python | Alert library used in-process by tracking (EWMA, rules, sinks) |
| `display_service` | Python | FastAPI dashboard: MJPEG camera wall, alerts, lifecycle control |
| `control_service` | Python | Fleet lifecycle facade over systemd/compose + runtime camera plane |

[Data Flow](#workinghigh-level-data-flow)

### Ports

| Service | gRPC | Metrics | Other |
|---|---|---|---|
| ingest | `:50053` (admin) | `:9101` | — |
| inference | `:50051` | `:9102` | — |
| tracking | `:50052` | `:9103` | — |
| display | — | `:8088/metrics` | `:8088` dashboard |
| mediamtx | — | — | `:8554` RTSP |
| Prometheus | — | — | `:9090` |

Every service exposes `/metrics` on its **own** HTTP listener. This is not an oversight — a
`grpc::Server` dispatches HTTP/2 + protobuf against a registered service contract and has no hook
for plain HTTP/1.1, so a scrape endpoint cannot share the gRPC socket. `METRICS_ADDR=off` disables
the endpoint per service.

### Detection classes

`0=Product`, `1=Product-Picked`, `2=Regular`, `3=Shoplifting`. Only class 3 currently contributes to
the suspicion signal.

---

## My setup / Development Stack

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS (kernel 6.8) |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM |
| Driver | 580.159.03 |
| Compiler | GCC 12.3.0, C++17 |
| CMake | 3.22.1 |
| Python | 3.10.12 |

### Dependencies

The GPU stack is **vendored under `dependency/`** rather than installed system-wide, so the ONNX
Runtime/TensorRT versions are pinned per-project and don't fight whatever CUDA the distro has. The
run scripts and systemd units put these on `LD_LIBRARY_PATH` explicitly, both resolving the paths from
`scripts/dependencies.conf` so they cannot drift apart. Nothing has to live under `dependency/` —
point the conf at `/opt` or a shared mount if that is where your stack is.

| Dependency | Version | Location |
|---|---|---|
| CUDA | 12.8.0 | `dependency/cuda_12.8.0` |
| cuDNN | 9.23.2.1 | `dependency/cudnn-9.23.2.1` |
| TensorRT | 10.8.0.43 | `dependency/TensorRT-10.8.0.43` |
| ONNX Runtime (GPU) | 1.22.0 | `dependency/onnxruntime-linux-x64-gpu-1.22.0` |
| gRPC + protobuf | 1.78.1 / protoc 31.1 | `dependency/grpc` → `~/.local` |
| prometheus-cpp | — | `dependency/prometheus-cpp/_install` |
| Prometheus server | 3.13.2 | `dependency/prometheus-3.13.2.linux-amd64` |
| yaml-cpp | 0.9.0 (tag `yaml-cpp-0.9.0`) | `dependency/yaml-cpp/_install` |
| OpenCV (C++) | 4.8.0 | `/usr/local/lib/cmake/opencv4` |
| MediaMTX | v1.16.2 | `rtsp_server/mediamtx` |
| FFmpeg | 4.4.2 | system |

Python packages are pinned in `requirements.txt`:

## Prerequisites and Dependencies

```bash
git clone https://github.com/Rohan7501/Multi-Camera-Security-System--Theft-Detection.git
cd Multi-Camera-Security-System--Theft-Detection
mkdir dependency
export SEC_SYS_ROOT_DIR=$PWD
```

It recommended to add `SEC_SYS_ROOT_DIR` to `~/.bashrc` as all scripts depend on it 
```bash
nano ~/.bashrc
```

Paste at the end of `.bashrc` file
```bash
export SEC_SYS_ROOT_DIR=/path/to/repo
```

Reload changes
```bash
source ~/.bashrc
echo $SEC_SYS_ROOT_DIR # Should print path to repo root
```

These are the **only** commands in the bring-up that need `sudo` —
`scripts/install_dependencies.sh` deliberately contains none, so it can run unprivileged and installs
everything either into `dependency/` or into `~/.local`. 

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    autoconf libtool \
    zlib1g-dev libssl-dev \
    curl ca-certificates
```
Python dependencies

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest and ruff
```

OpenCV(C++) is expected to already be installed at `/usr/local/lib/cmake/opencv4`.

Install build dependencies
```bash
scripts/install_dependencies.sh
```

| Env | Effect |
|---|---|
| `SEC_SYS_ROOT_DIR` | **required** — repo root; the script exits if unset |
| `JOBS` | parallel compile jobs (default: ~1 per 2 GB of RAM, capped at `nproc`) |
| `FORCE=1` | rebuild dependencies that are already installed |

Each step short-circuits on its installed-config marker, so a re-run after a failure resumes rather
than rebuilding gRPC from scratch.

gRPC needs to be installed user-wide because of the other dependencies
it pulls in, therefore `install` present separately

```bash
cmake --install "dependency/grpc/cmake/build"
```

The prebuilt GPU stack is **not** covered by the script — download the CUDA/cuDNN/TensorRT tarballs
from NVIDIA and the ONNX Runtime tarball from Microsoft into `dependency/`, using the exact
version directory names in the table above. 
If you already have the stack installed you can use that as well.
Paste the directories in `scripts/dependencies.conf`

Usefull Links
```bash
# CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
chmod +x cuda_12.8.0_570.86.10_linux.run
./cuda_12.8.0_570.86.10_linux.run --toolkit \
  --toolkitpath="$SEC_SYS_ROOT_DIR/dependency/cuda_12.8.0" --no-drm --override --silent

# TensorRT
https://developer.nvidia.com/tensorrt/download/10x
→ TensorRT 10.8 GA → TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8.tar.gz

# cuDNN
https://developer.nvidia.com/cudnn-downloads
→ Linux x86_64 → Tarball → cudnn-linux-x86_64-9.23.2.1_cuda12-archive.tar.xz

# Onnx runtime
https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0
→ onnxruntime-linux-x64-gpu-1.22.0.tgz
```

Place your trained YOLOv8 export at `models/best.onnx`.
You can download the models from: https://drive.google.com/drive/folders/1NQmzW6VyQ1RuNGcYgWJcdRNy96g3iORh?usp=drive_link

---

## Build

### 1. Compile the protos

Regenerate after any change to `proto/services.proto`. It takes **two** invocations — one per language
— because `--plugin=protoc-gen-grpc=` binds a single plugin to the name `grpc`, and C++ and Python need
different ones:

```bash
for svc in tracking_service control_service; do
  protoc -I proto \
    --python_out=services/$svc \
    --grpc_out=services/$svc \
    --plugin=protoc-gen-grpc=$(which grpc_python_plugin) \
    proto/services.proto
done
```

C++ gets `services.pb.{cc,h}` and `services.grpc.pb.{cc,h}`; Python gets `services_pb2.py` and
`services_pb2_grpc.py`. The Python services use **flat imports** (`import services_pb2`), which is why
the stubs live beside the code and why each service must be run from its own directory.

The C++ side is also generated **automatically** by CMake into `build/` as part of the
`inference_proto` target, and that copy is what actually compiles. 

### 2. Compile the project

```bash
cmake -S . -B build
cmake --build build -j
```

Build type defaults to `RelWithDebInfo`. Only `common`, `services/ingest_service` and
`services/inference_service` are in the CMake build; the Python services are not compiled.

> **Don't `rm -rf build`.** It discards the protobuf/gRPC generation cache and forces a slow full
> regenerate. `cmake --build build -j` is incremental and sufficient.

### 3. Run the tests

Both languages run under one `ctest` invocation — the Python suite is registered as a ctest case, so
there is a single command to gate on:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

```
1/3 Test #1: frame_queue ......................   Passed
2/3 Test #2: rtsp_reader ......................   Passed
3/3 Test #3: python_unit ......................   Passed
```

| Test | Covers |
|---|---|
| `frame_queue` | bounded queue: drop-oldest, cumulative drop count, shutdown wakes every waiter |
| `rtsp_reader` | dead/garbage camera URLs must not hang, crash or leak threads |
| `python_unit` | suspicion EWMA + hysteresis, rules, engine dedupe/cooldown, frame age, sinks |

A detector smoke test needs a GPU and a model, it tests EP and CPU/GPU nms performace:
It builds a tensorRT engine, expect it to take 10 mins on the slower side.
Export CUDA, cuDNN, TensorRT and onnx-runtime path to run `onnx_test`. 
Rest of the cases, where those paths are required, are handled by the scripts
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/CUDA/:path/to/cudnn/lib:path/to/tensorrt/lib:path/to/onnxruntime/lib
./build/services/inference_service/onnx_test --raw models/best.onnx --nms models/best_nms.onnx --image tests/frame_666_640x640.jpg
```

<!-- The end-to-end benchmark, `tests/benchmark_pipeline.py`, also lives in `tests/` but is not part of
`ctest`: it needs the GPU, the model and live cameras. See [Benchmarking](#benchmarking). -->

---

## Simulating RTSP streams

There are no real cameras in development. MediaMTX acts as the RTSP server and FFmpeg loops a video
file into it once per camera, which is indistinguishable from a real feed to ingest.
Check readme.md at `/rtsp_server` for download

Start the RTSP server:

```bash
./rtsp_server/mediamtx          # listens on :8554
```

Publish 8 looping camera feeds matching `config.yaml` (`cam1` … `cam8`):

```bash
VIDEO=~/Downloads/videoplayback.mp4
for i in $(seq 1 8); do
  ffmpeg -re -stream_loop -1 -i "$VIDEO" -c copy -f rtsp rtsp://localhost:8554/cam$i \
    -loglevel error &
done
```

Verify a stream is live before starting the pipeline:

```bash
ffprobe rtsp://localhost:8554/cam1
```

Cameras are declared in `config.yaml`, which ingest reads at startup:

```yaml
cameras:
  - id: cam1
    url: rtsp://localhost:8554/cam1
```

To stop the simulated cameras, kill the FFmpeg publishers by port rather than `pkill -f ffmpeg`
(a `pkill` pattern can match your own shell):

```bash
kill $(pgrep -f 'rtsp://localhost:8554/cam')
```

---

## Running

Start order is always **display → tracking → inference → ingest**.

Display comes first because it is the front door — it runs standalone, shows placeholders until
frames arrive, and hosts the buttons that start everything else. The remaining three are ordered by
who dials whom: inference is tracking's gRPC client, and ingest is inference's, so each server must
be listening before its client starts.

> **Readiness, not just ordering.** A client that starts before its server is up will fail its first
> dial. `systemctl start` returns when the process is *exec'd*, not when its port is bound, so start
> ordering alone is not enough — options 1 and 3 handle this differently, see below.

All three options need `FRAME_TRANSPORT` to be **identical** across ingest, inference and tracking.
The C++ `checkTransport` handshake aborts a peer on mismatch rather than letting it read garbage.

### Installing the systemd units — do this first for options 1 and 3

Option 2 runs the binaries directly and needs none of this. **Options 1 and 3 both drive
`systemctl --user`**, so the units and their env files must be in place before either will work.

Units are **user units** in `~/.config/systemd/user/`, not system units. That matters: the shm segment
is created `0600`, so producer and consumer must share an owner, and `--user` units already run as the
invoking user with no `User=`/`Group=` needed.

The units and env files ship in `deploy/` as **templates** (`*.in`) and are rendered for your checkout
by the installer. Rendering is **all** it does — it installs nothing, never runs `sudo`, and never
talks to systemd:

```bash
scripts/install_systemd.sh --dry-run     # print what would be rendered
scripts/install_systemd.sh               # render into deploy/rendered/
```

That leaves eight files in the gitignored `deploy/rendered/` — four `.env` and four `.service`.

```bash
# 1. env files, as root (units won't start without them)
sudo install -d /etc/edge-ai
sudo install -m 644 deploy/rendered/*.env /etc/edge-ai/

# 2. user units, then pick them up
install -d ~/.config/systemd/user
install -m 644 deploy/rendered/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

The script prints these same commands when it finishes, with your paths already filled in.

Copying and reloading stay manual on purpose. One step needs root and the other changes the state of
a box that may be running the fleet — neither should happen as a side effect of regenerating some
files. It also means you can diff `deploy/rendered/` against what is already installed before
overwriting anything.

The templating is not optional. systemd expands `${VAR}` only in `ExecStart=` *arguments* — never in
`WorkingDirectory=`, `EnvironmentFile=`, or the `ExecStart=` binary path, all of which must be
literal absolute paths. So `SEC_SYS_ROOT_DIR` cannot be referenced from a unit file directly; it is
substituted at install time instead. Override with `SEC_SYS_ROOT_DIR`, `EDGE_AI_ETC` or `UNIT_DIR`.

Re-run the installer after moving the repo or editing anything in `deploy/` — the rendered paths are
absolute and go stale.

> **First run — use Option 2.** The first start builds the TensorRT engine, so the inference service
> takes 5–10 minutes to come up. That is long enough to trip the readiness gates in options 1 and 3;
> run it once from a terminal where you can watch it, then switch to whichever option you prefer.

### Option 1 — script + systemd facade (recommended)

Needs the units installed — see [Installing the systemd units](#installing-the-systemd-units--do-this-first-for-options-1-and-3)
above.

Start the display service, then drive the whole pipeline from the browser. The display service calls
`control_service.FleetController`, which delegates to `systemctl --user` in dependency order **and
gates on readiness** — after starting each service it blocks until that service's port is accepting
before starting the peer that dials it (default 20 s, `STARTUP_READY_TIMEOUT`).

```bash
systemctl --user start edge-display      # or: scripts/run_display.sh
xdg-open http://localhost:8088/
```

Then use the control shell:

| Button | Endpoint | Behaviour |
|---|---|---|
| Start system | `POST /start-up` | tracking → inference → ingest, readiness-gated |
| Stop system | `POST /shutdown` | ingest → inference → tracking (reverse) |
| Restart | `POST /restart` | full stop-all then start-all |
| Status | `GET /status` | per-service `is-active` |

Or from the terminal:

```bash
curl -X POST localhost:8088/start-up
curl -X POST localhost:8088/shutdown
curl localhost:8088/status
```

`LIFECYCLE_BACKEND` (in `/etc/edge-ai/display.env`) selects the backend: `systemd-user` (matches the
`--user` units used here), `systemd`, `compose`, or `dryrun` to log the commands without touching
anything.

**Restart cycles the whole fleet, not one unit.** A per-unit `systemctl restart` would leave
inference and ingest holding a connection to a process that no longer exists.

### Option 2 — scripts only (no systemd)

Four terminals, in order. Each script exports the environment its binary needs, so nothing depends on
your shell state.

```bash
# terminal 1 — display
scripts/run_display.sh

# terminal 2 — tracking
scripts/run_tracker.sh

# terminal 3 — inference   (waits for nothing; start it after tracking is listening)
scripts/run_inference.sh

# terminal 4 — ingest
scripts/run_ingest.sh
```

Bring the cameras up **before** terminal 4 — `run_ingest.sh` runs the ingest binary and nothing else.
Start MediaMTX and the FFmpeg publishers as shown in [Simulating RTSP streams](#simulating-rtsp-streams)

`run_inference.sh` sets `LD_LIBRARY_PATH` for the vendored CUDA/cuDNN/TensorRT/ORT libraries — the
inference binary will not start without it.

Watch for `InferenceWorker pool started` and `TrackingClient -> 127.0.0.1:50052` in the inference
terminal — that pair means the pipeline is wired up.

This option gives you stdout in front of you, which is the reason to prefer it while debugging.

### Option 3 — systemd only

Needs the units installed — see [Installing the systemd units](#installing-the-systemd-units--do-this-first-for-options-1-and-3)
above. No facade here: you drive `systemctl` yourself and own the ordering.

Start in order:

```bash
systemctl --user start edge-display
systemctl --user start edge-tracking
systemctl --user start edge-inference
systemctl --user start edge-ingest
```

Stop in reverse:

```bash
systemctl --user stop edge-ingest edge-inference edge-tracking edge-display
```

Enable at login:

```bash
systemctl --user enable edge-display edge-tracking edge-inference edge-ingest
loginctl enable-linger $USER      # so units survive logout
```

The units carry `After=`/`Before=` ordering (`tracking → inference → ingest`), but **`After=` orders
starts, not readiness**. Starting all four at once can still race; `Restart=on-failure` with
`RestartSec=2` covers a too-early start by retrying. Option 1's explicit port gate is the
deterministic version of this.

Launch-time configuration lives in `/etc/edge-ai/{display,tracking,inference,ingest}.env`, read via
`EnvironmentFile=`. These are rendered by the control service and applied on the next restart —
systemd env-file syntax is bare `KEY=VALUE`, with no quoting, expansion or `export`.

#### Stopping a service that keeps restarting

`Restart=on-failure` means killing the process just makes systemd start a new one. Ask systemd to
stop it instead:

```bash
systemctl --user stop edge-ingest         # honors the stop; won't be restarted
systemctl --user disable edge-ingest      # don't start at login
systemctl --user mask edge-ingest         # refuse to start at all, even as a dependency
systemctl --user reset-failed edge-ingest # clear a failed state after too many restarts
```

---

## Observability

```bash
scripts/run_prometheus.sh                 # UI + API on :9090, 7d retention
curl localhost:9102/metrics               # scrape one service by hand
```

Prometheus synthesises `up{service=…}` per target, which is the fleet liveness signal — no health
endpoint required. Scrape config is in `deploy/prometheus.yml`.

### Per-hop latency

Two timestamps travel with every frame, with deliberately different lifetimes. `gTimestampNs` is
**re-stamped at each hop** with that stage's receipt time, so a service can measure the one segment
it just completed. `gCaptureTimestampNs` is **set once by ingest at the RTSP read and never touched
again**, so end-to-end frame age survives the whole pipeline.

| Metric | Exposed by | Measures |
|---|---|---|
| `ingest_module_latency_seconds{stream}` | inference `:9102` | ingest RTSP read → inference receipt |
| `inference_module_latency_seconds{stream}` | tracking `:9103` | inference receipt → tracking receipt |
| `pipeline_frame_age_seconds{stream}` | tracking `:9103` | **end-to-end**: RTSP read → tracking |

On a 15 fps synthetic stream these measure 0.77 ms, 5.20 ms and 5.96 ms — the hops sum exactly to the
end-to-end age, which is the invariant to check if you add a hop.

Useful queries:

```promql
histogram_quantile(0.50, sum(rate(inference_latency_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(inference_latency_seconds_bucket[5m])) by (le, stream))
rate(inference_frames_total[1m])
inference_tracking_stream_up
```

The `by (le)` is mandatory when aggregating — `histogram_quantile` needs the bucket boundary label to
survive the `sum`.

**Cardinality rule:** label by `stream` (bounded camera set) and `class_id` (4 classes), never by
`track_id` or `frame_id` (unbounded).

### Logs

systemd captures each binary's stdout and stderr to the journal. The C++ services log with
`std::endl`, which flushes per line, so output appears live:

```bash
scripts/logs.sh                    # follow all edge units, interleaved
scripts/logs.sh inference          # just edge-inference
scripts/logs.sh inference ingest   # a subset
journalctl --user -u edge-inference -f
```

<!-- ---

## Benchmarking

`tests/benchmark_pipeline.py` measures the whole pipeline under its real workload, the cameras in
`config.yaml`. It records an idle baseline and discards a warmup. It opens the measurement window only
once every camera is flowing at a steady rate, and it records every frame's latency rather than
estimating it.

```bash
python3 tests/benchmark_pipeline.py --duration 5m               # laptop / discrete GPU
python3 tests/benchmark_pipeline.py --duration 5m --jetson      # Jetson: + temperature and power
python3 tests/benchmark_pipeline.py --attach --duration 60      # services already running (systemd)
NUM_THREADS=6 python3 tests/benchmark_pipeline.py               # services inherit the environment
```

By default it starts the pipeline itself through `scripts/run_*.sh` (tracking → inference → display →
ingest) and stops it afterwards. Run it from the shell you'd use for those scripts (on the Jetson,
with the venv on `PATH`), with the cameras already up. It refuses to start if a pipeline port is
taken. Results go to `data/benchmarks/<timestamp>/`: `report.txt`, `results.json`, `frames.csv` (every
frame), `timeline.csv` (per second), the raw `tegrastats`/`nvidia-smi`/`jtop` logs, and each
service's log.

| Flag | Default | Meaning |
|---|---|---|
| `--duration` | `60` | Measurement window: `90`, `90s`, `5m`, `1h` (startup and warmup come on top) |
| `--jetson` | off | Add temperatures, power rails, GPU/EMC load and clocks, and per-process GPU memory |
| `--warmup` | `15` (`0` with `--attach`) | Discarded before the steady-state gate |
| `--idle` | `5` | Idle baseline before launch, for the memory and power deltas |
| `--interval` | `1` | Sampling interval |
| `--attach` | off | Measure services that are already running instead of launching them |
| `--no-display` | off | Leave the display service out |
| `--out` | `data/benchmarks/<timestamp>` | Results directory |

| Section | Numbers | Source |
|---|---|---|
| Latency | End-to-end and per hop: mean, p50/p90/p95/p99/p99.9, max; per camera; drift test | Every frame, from tracking's `LATENCY_TRACE` |
| Throughput | Captured / inferred / tracked fps, every drop counter, queue depth | Prometheus counters over the window |
| Memory, CPU | Per service: CPU, threads, RSS split into anon / file / shmem, PSS, peak, heap drift | `/proc/<pid>/status`, `smaps_rollup` |
| GPU | Per-process and device memory, utilisation | `nvidia-smi`; `jtop` on a Jetson |
| Jetson | Power per rail against idle, mJ/frame, FPS/W, temperature per zone, clocks | `tegrastats`, `jtop`, `nvpmodel`, sysfs |

Reading the report:

- **Percentiles are exact when the benchmark launched the pipeline**: tracking writes one line per
  frame. With `--attach` they come from Prometheus buckets and are marked ESTIMATED. Around the
  operating point the buckets are 10/25/50 ms wide, and in testing the estimates overstated p90 by
  about 50%. For exact numbers, start tracking with `LATENCY_TRACE=<file>`; `--attach` finds the file
  through its environment.
- **Read the tail together with the drops.** A dropped frame has no latency sample.
- **Pipeline footprint is PSS plus GPU memory.** RSS counts shared pages (the shm ring, the CUDA
  libraries) in full in every process, so RSS summed across services overcounts. A leak shows up as
  *anon drift*. On a discrete GPU the two are separate pools: system RAM plus the card's VRAM, which
  `nvidia-smi` reports. On a Jetson both live in the same RAM. Most GPU allocations appear in no
  process's RSS, which is why they come from `jtop`. The buffers mapped into the process (`/dmabuf:`
  in smaps, about 186 MB for inference) also show up in PSS, so they are taken out of the PSS side
  and counted once.
- **Jetson clocks:** unless `sudo jetson_clocks` ran first, DVFS moves the clocks during the run, and
  the report says so. A "still warming" line means temperatures were still rising at the end: run
  longer for sustained numbers.
- **One copy of each service.** gRPC sets `SO_REUSEPORT`, so a second copy of a service binds the same
  port without an error and takes part of the traffic. `--attach` refuses to measure if it finds one. -->

---

## Configuration

| Variable | Default | Applies to | Meaning |
|---|---|---|---|
| `FRAME_TRANSPORT` | `shm` | ingest, inference, tracking | `shm` ring or `grpc` inline; **must match across all three** |
| `TRACKING_PIXELS` | `0` | inference, tracking | Does the tracker consume pixels? ByteTrack does not |
| `TRACKING_ADDR` | `127.0.0.1:50052` | inference | TrackingService endpoint |
| `INFERENCE_ADDR` | `localhost:50051` | ingest | InferenceService endpoint |
| `INGEST_ADMIN_ADDR` | `127.0.0.1:50053` | ingest | Runtime camera-control plane |
| `METRICS_ADDR` | per service | all | Scrape endpoint; `off` disables |
| `DISPLAY_PUBLISH` | `1` | tracking | Publish detections to the display feed |
| `DISPLAY_DETECTIONS` | `0` | tracking | In-process `cv2` window — **off under systemd** (headless) |
| `EVENT_DATA_DIR` | `<repo>/data` | tracking, display | Alerts JSON + SQLite root |
| `INFERENCE_MAX_AGE_MS` | `40` | inference | Safety net: skip a frame already this old (capture age) when a worker takes it; `0` disables |
| `TRACKING_MAX_AGE_MS` | `80` | tracking | Safety net: skip a frame already this old when it arrives; `0` disables |
| `OPENBLAS_NUM_THREADS` | `1` | tracking | numpy/scipy BLAS threads. Keep `1`: idle OpenBLAS threads busy-wait (set by `run_tracker.sh`, the unit and `main.py`) |
| `NUM_THREADS` | `4` | inference | Inference worker-pool size |
| `LATENCY_TRACE` | unset | tracking | Write one CSV line per tracked frame to this file, for the [benchmark](#benchmarking) |
| `LIFECYCLE_BACKEND` | `systemd-user` | display | `systemd` / `systemd-user` / `compose` / `dryrun` |
| `DISPLAY_FPS` | `15` | display | MJPEG cap per viewer |
| `STARTUP_READY_TIMEOUT` | `20` | display | Seconds to wait for a port before starting its peer |

More inference workers (`NUM_THREADS`) raise throughput but also per-frame latency, since
concurrent `Ort::Session::Run` calls contend for the GPU — read `rate(inference_frames_total[1m])`
alongside the latency percentiles, not either alone.

---

## Dashboard

`http://localhost:8088/` is the control shell (Start / Stop / Restart / Status).
`http://localhost:8088/dashboard-display` is the video wall plus recent alerts.

All cameras are composited into **one** MJPEG stream at `/wall`. This is not a stylistic choice:
browsers allow only ~6 concurrent HTTP/1.1 connections per host and an MJPEG response never ends, so
with one `<img>` per camera the 7th and 8th tiles never get a connection *and* the alerts poll
starves behind the saturated six. One mosaic is one connection at any camera count. The grid is
`ceil(sqrt(n))` wide with 480×270 cells, so 8 cameras render as 3×3 at 1440×810.

---

## Repository layout

```
common/            shared C++/Python: shm ring, types, metrics, config, logging
proto/             services.proto — the compiled gRPC contract
services/          the six services (see table above)
scripts/           run_*.sh launchers, logs.sh, Prometheus, ONNX tooling
deploy/            prometheus.yml, systemd/*.service.in and env/*.env.in templates
rtsp_server/       vendored MediaMTX
tests/             pytest suite + C++ unit tests, all registered with ctest
models/            best.onnx (runtime), best.pt (source), best.engine (TensorRT)  [gitignored]
dependency/        vendored CUDA / cuDNN / TensorRT / ORT / gRPC / prometheus     [gitignored]
```

---

## In progress

- [ ] Optimizing for an edge device — Jetson (aarch64)
- [ ] Docker backend
- [ ] Adding and removing cameras on a running system

---

## License

MIT — see [LICENSE](LICENSE).
