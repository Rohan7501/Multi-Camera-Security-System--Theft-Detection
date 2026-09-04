"""Shared pytest setup.

The Python services use flat imports (`from suspicion import TrackState`), so
each service directory has to be on sys.path before its modules resolve. Order
matters: several services have their own main.py, and the first match on the
path wins.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICES = REPO_ROOT / "services"

# event_service first -- it owns the modules under test here. Appending rather
# than inserting keeps pytest's own rootdir ahead of us.
for svc in ("event_service", "tracking_service"):
    p = str(SERVICES / svc)
    if p not in sys.path:
        sys.path.append(p)
