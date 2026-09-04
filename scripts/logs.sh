#!/usr/bin/env bash
# Live-tail a service's stdout/stderr in this terminal for debugging.
#
# The systemd units capture each binary's stdout+stderr to the journal (the
# default StandardOutput=journal). The C++ services log with `std::cout << ...
# << std::endl` and `std::cerr`, both of which flush per line, so lines appear
# in the journal live -- no buffering workaround needed.
#
#   scripts/logs.sh                     # follow ALL edge units, interleaved
#   scripts/logs.sh inference           # follow one (edge-inference)
#   scripts/logs.sh inference ingest    # follow a subset
#   scripts/logs.sh -- -n 500 ...       # extra args after `--` go to journalctl
#
# --user because the units are installed as `systemctl --user` units; drop it
# (SCOPE=) if you installed them as system units.
set -euo pipefail

SCOPE="${SCOPE---user}"          # override: SCOPE= scripts/logs.sh  (system units)
ALL=(edge-inference edge-ingest edge-tracking edge-display)

names=()
extra=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --) shift; extra=("$@"); break ;;      # pass the rest straight to journalctl
    *)  names+=("$1"); shift ;;
  esac
done

units=()
if [ "${#names[@]}" -eq 0 ]; then
  units=("${ALL[@]}")
else
  for s in "${names[@]}"; do units+=("edge-${s#edge-}"); done   # accept 'ingest' or 'edge-ingest'
fi

sel=()
for u in "${units[@]}"; do sel+=(-u "$u"); done

# -f follow, -n 100 seed with recent history, short-precise = microsecond stamps.
exec journalctl ${SCOPE:+$SCOPE} "${sel[@]}" -f -n 100 --output=short-precise "${extra[@]}"
