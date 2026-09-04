"""Display service: live annotated video (MJPEG) + alerts + Prometheus /metrics.

Option-A standalone service. Reads raw frames from the shm ring and detections
from the tracking feed (see frames.py), draws, and streams JPEG to the browser.
Also serves recent alerts from the event_service SQLite (read-only) and exposes
its own health on /metrics for Prometheus to scrape.

Run:  cd services/display_service && uvicorn main:app --host 0.0.0.0 --port 8088
(or scripts/run_display.sh). Requires the pipeline running with FRAME_TRANSPORT=shm
and tracking started with DISPLAY_PUBLISH=1.

Scaling note: each MJPEG viewer holds one worker thread for the life of the
connection (sync generator). Fine for a handful of operator screens; front with
a fan-out proxy for many viewers.
"""
import logging
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from frames import FrameJoiner
from metrics import FRAMES_SERVED, FRAME_AGE, SHM_MISS, VIEWERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("display")

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_DATA_DIR = Path(os.getenv("EVENT_DATA_DIR", _REPO_ROOT / "data"))
TARGET_FPS = float(os.getenv("DISPLAY_FPS", "10"))
# Label for the mosaic viewer gauge. One extra label value, not one per camera,
# so it respects the bounded-label rule.
_WALL = "(wall)"
_CONTROL_DIR = str(_REPO_ROOT / "services" / "control_service")

app = FastAPI(title="edge-ai display service")
joiner = FrameJoiner()

# ---- lifecycle facade (control_service.FleetController) ---------------------

# Start dependency order: tracking has no reconnect and must be up before
# inference dials it; inference (the gRPC server) must be up before ingest (the
# client). See each service's CLAUDE.md.
STARTUP_ORDER = ("tracking", "inference", "ingest")
# Tear down in the reverse order: drop the clients (ingest, then inference)
# before the servers they dial, so nothing is left talking to a dead peer.
SHUTDOWN_ORDER = tuple(reversed(STARTUP_ORDER))

# Readiness gate. `systemctl start` returns when the process is exec'd, not when
# its port is listening -- so we wait for the port a downstream peer will dial
# before starting that peer. Maps service -> the (host, port) it exposes.
# See README.md.
def _endpoint(env_var: str, host: str, port: int):
    hp = os.getenv(env_var, "")
    if hp:
        h, sep, p = hp.rpartition(":")
        if sep:
            return (h or host, int(p))
    return (host, port)

READY_ENDPOINT = {
    "tracking":  _endpoint("TRACKING_ADDR",  "127.0.0.1", 50052),  # inference dials this
    "inference": _endpoint("INFERENCE_ADDR", "127.0.0.1", 50051),  # ingest dials this
    # ingest exposes only the admin RPC (:50053); nothing downstream waits on it.
}
READY_TIMEOUT = float(os.getenv("STARTUP_READY_TIMEOUT", "20"))


def _wait_ready(host: str, port: int, timeout: float) -> bool:
    """Block until (host, port) accepts a TCP connection, or timeout. A successful
    connect means the gRPC listener is bound -- enough to let the peer dial in."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


_fleet = None


def fleet():
    """Lazily build the FleetController. Backend from LIFECYCLE_BACKEND:
    systemd (default) | systemd-user | compose | dryrun (log commands only)."""
    global _fleet
    if _fleet is None:
        if _CONTROL_DIR not in sys.path:
            sys.path.append(_CONTROL_DIR)          # append: our modules keep priority
        from lifecycle import SystemdBackend, ComposeBackend
        from fleet_controller import FleetController
        mode = os.getenv("LIFECYCLE_BACKEND", "systemd-user")
        if mode == "dryrun":
            backend = SystemdBackend(dry_run=True)
        elif mode == "systemd-user":
            backend = SystemdBackend(user=True)
        elif mode == "compose":
            backend = ComposeBackend(os.getenv("COMPOSE_FILE", "docker-compose.yml"))
        else:
            backend = SystemdBackend()
        _fleet = FleetController(backend)
        log.info("lifecycle backend: %s (LIFECYCLE_BACKEND=%s)", type(backend).__name__, mode)
    return _fleet


# ---- alerts (read-only view of the event_service SQLite) -------------------

def recent_alerts(limit: int = 25) -> list:
    db = EVENT_DATA_DIR / "alerts.db"
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, wall_ts, stream_id, track_id, rule, score, frame_id, "
            "       frame_age_s "
            "FROM alerts WHERE suppressed=0 ORDER BY wall_ts DESC LIMIT ?",
            (limit,)).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        log.warning("alerts read failed: %s", e)
        return []


# ---- MJPEG ------------------------------------------------------------------

def mjpeg(stream_id: str):
    VIEWERS.labels(stream_id).inc()
    try:
        period = 1.0 / TARGET_FPS
        while True:
            t0 = time.monotonic()
            jpg, age, had = joiner.render(stream_id)
            FRAMES_SERVED.labels(stream_id).inc()
            if not had:
                SHM_MISS.labels(stream_id).inc()
            if age is not None:
                FRAME_AGE.labels(stream_id).set(age)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        # Client disconnect closes the generator -> this runs.
        VIEWERS.labels(stream_id).dec()


def mosaic():
    """All cameras composited into ONE MJPEG stream.

    One connection for any camera count; list refreshes every second. README.md.
    """
    VIEWERS.labels(_WALL).inc()
    try:
        period = 1.0 / TARGET_FPS
        streams, next_refresh = joiner.streams(), 0.0
        while True:
            t0 = time.monotonic()
            if t0 >= next_refresh:
                streams = joiner.streams()
                next_refresh = t0 + 1.0
            jpg, stats = joiner.render_wall(streams)
            for sid, age, had in stats:
                FRAMES_SERVED.labels(sid).inc()
                if not had:
                    SHM_MISS.labels(sid).inc()
                if age is not None:
                    FRAME_AGE.labels(sid).set(age)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        VIEWERS.labels(_WALL).dec()


# ---- routes -----------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/streams")
def streams():
    return {"streams": joiner.streams()}

@app.get("/alerts")
def alerts(limit: int = 25):
    return JSONResponse(recent_alerts(limit))

@app.get("/stream/{stream_id}")
def stream(stream_id: str):
    return StreamingResponse(mjpeg(stream_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/wall")
def wall():
    """The whole camera wall as ONE MJPEG stream (see mosaic() for why)."""
    return StreamingResponse(mosaic(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

def _run_lifecycle(order, verb: str, wait: bool = False) -> list:
    """Apply one lifecycle verb across `order`, best-effort (one failure is
    reported, not fatal to the rest). When `wait`, block after each start until
    that service's port is accepting before moving to its downstream peer (the
    peer dials it once, no reconnect). Returns a per-service result list."""
    fc = fleet()
    results = []
    for svc in order:
        try:
            getattr(fc, verb)(svc)          # fc.start / fc.stop / fc.restart
            entry = {"service": svc, "ok": True, "detail": f"{verb} issued"}
        except Exception as e:
            log.warning("%s %s failed: %s", verb, svc, e)
            results.append({"service": svc, "ok": False, "detail": str(e)})
            continue
        if wait and svc in READY_ENDPOINT:
            host, port = READY_ENDPOINT[svc]
            if _wait_ready(host, port, READY_TIMEOUT):
                entry["detail"] += f"; {host}:{port} ready"
            else:                            # start the rest anyway, but flag it
                entry["ok"] = False
                entry["detail"] += f"; {host}:{port} NOT listening after {READY_TIMEOUT:g}s"
                log.warning("readiness timeout: %s at %s:%d", svc, host, port)
        results.append(entry)
    return results


@app.post("/start-up")
def start_up():
    """Bring the pipeline up via the control_service facade, in dependency order,
    waiting for each server to accept before starting the peer that dials it."""
    return {"backend": type(fleet().backend).__name__, "verb": "start",
            "order": list(STARTUP_ORDER),
            "results": _run_lifecycle(STARTUP_ORDER, "start", wait=True)}

@app.post("/shutdown")
def shutdown():
    """Stop the pipeline (reverse dependency order). This tells systemd to stop
    the units and stay stopped -- it does not fight the Restart= policy."""
    return {"backend": type(fleet().backend).__name__, "verb": "stop",
            "order": list(SHUTDOWN_ORDER),
            "results": _run_lifecycle(SHUTDOWN_ORDER, "stop")}

@app.post("/restart")
def restart():
    """Full-system restart: stop everything, then start it clean. A per-unit
    `systemctl restart` would leave inference/ingest holding a dead connection
    (their gRPC clients have no reconnect), so cycle the whole fleet instead."""
    down = _run_lifecycle(SHUTDOWN_ORDER, "stop")
    up = _run_lifecycle(STARTUP_ORDER, "start", wait=True)
    return {"backend": type(fleet().backend).__name__, "verb": "restart",
            "phases": {"stop": down, "start": up}}

@app.get("/status")
def status():
    """Live is-active state per service (for the control shell / debugging)."""
    fc = fleet()
    try:
        return {"backend": type(fc.backend).__name__,
                "services": [vars(s) for s in fc.status()]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
def index():
    return f"""<!doctype html><meta charset=utf-8>
<title>edge-ai control</title>
<style>
 body{{font-family:system-ui;margin:2rem;background:#111;color:#eee}}
 button{{font-size:1rem;padding:.6rem 1.2rem;cursor:pointer}}
 pre{{background:#000;border:1px solid #333;padding:1rem;white-space:pre-wrap}}
 a{{color:#6cf}}
</style>
<h1>edge-ai control</h1>
<p>Bring the pipeline up (tracking &rarr; inference &rarr; ingest), then open the dashboard.</p>
<button onclick="call('/start-up')">Start system</button>
<button onclick="call('/restart')">Restart system</button>
<button onclick="call('/shutdown')">Stop system</button>
<button onclick="call('/status','GET')">Status</button>
<a href="/dashboard-display" style="margin-left:1rem">Open dashboard &rarr;</a>
<pre id=out>idle</pre>
<script>
async function call(path, method='POST'){{
  const out = document.querySelector('#out');
  out.textContent = method + ' ' + path + ' ...';
  try {{
    const r = await fetch(path, {{method}});
    out.textContent = JSON.stringify(await r.json(), null, 2);
  }} catch(e) {{ out.textContent = 'error: ' + e; }}
}}
</script>"""

@app.get("/dashboard-display", response_class=HTMLResponse)
def dashboard_display():
    cams = joiner.streams()
    # ONE <img> for every camera. One tile per camera hit the browser's
    # ~6-connections-per-host cap and starved /alerts. See README.md.
    wall = ('<img id=wall src="/wall" alt="camera wall">' if cams else
            "<p>No streams publishing yet. Start tracking with DISPLAY_PUBLISH=1.</p>")
    links = " ".join(f'<a href="/stream/{s}">{s}</a>' for s in cams)
    return f"""<!doctype html><meta charset=utf-8>
<title>edge-ai display</title>
<style>
 body{{font-family:system-ui;margin:1rem;background:#111;color:#eee}}
 .grid{{display:flex;flex-wrap:wrap;gap:1rem}}
 #wall{{max-width:100%;border:1px solid #333;background:#000}}
 .links{{margin:.5rem 0;font-size:.9rem}} .links a{{color:#6cf;margin-right:.75rem}}
 table{{border-collapse:collapse;margin-top:1rem;width:100%}}
 td,th{{border:1px solid #333;padding:.25rem .5rem;font-size:.9rem;text-align:left}}
 .shoplift{{color:#ff5555;font-weight:600}}
</style>
<h1>edge-ai display</h1>
<div class=grid>{wall}</div>
<div class=links>single camera: {links}</div>
<h2>Recent alerts</h2>
<table id=alerts><thead><tr><th>time<th>stream<th>track<th>rule<th>score<th>frame<th>age</tr></thead>
<tbody></tbody></table>
<script>
async function poll(){{
  const r = await fetch('/alerts?limit=25'); const rows = await r.json();
  document.querySelector('#alerts tbody').innerHTML = rows.map(a=>{{
    const t = new Date(a.wall_ts*1000).toLocaleTimeString();
    // frame_age_s < 0 means the frame carried no capture stamp (older peer).
    const age = (a.frame_age_s == null || a.frame_age_s < 0)
      ? '&mdash;' : (a.frame_age_s*1000).toFixed(0)+' ms';
    return `<tr><td>${{t}}</td><td>${{a.stream_id}}</td><td>${{a.track_id}}</td>`+
           `<td class=shoplift>${{a.rule}}</td><td>${{a.score}}</td><td>${{a.frame_id}}</td>`+
           `<td>${{age}}</td></tr>`;
  }}).join('');
}}
poll(); setInterval(poll, 3000);
</script>"""