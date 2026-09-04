#!/usr/bin/env bash
# Prometheus server for the edge-ai fleet (scrapes each service's /metrics).
#
#   scripts/run_prometheus.sh          # UI + API on http://localhost:9090
#
# Retention is capped for an edge box: the default 15d of every series is more
# disk than a store appliance wants. ~10MB/day at this series count, so 7d is
# well under 100MB. Data lands in data/prometheus (gitignored).
#
# This is the SERVER (scrape + TSDB + PromQL). The services themselves only
# expose /metrics -- they are targets, not Prometheuses.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROOT="$SEC_SYS_ROOT_DIR"
PROM="$DEP/prometheus-${PROM_VERSION:-3.13.2}.linux-amd64/prometheus"
RETENTION="${PROM_RETENTION:-7d}"
PORT="${PROM_PORT:-9090}"

mkdir -p "$ROOT/data/prometheus"
exec "$PROM" \
  --config.file="$ROOT/deploy/prometheus.yml" \
  --storage.tsdb.path="$ROOT/data/prometheus" \
  --storage.tsdb.retention.time="$RETENTION" \
  --web.listen-address="0.0.0.0:$PORT"