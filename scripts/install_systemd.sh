#!/usr/bin/env bash
# Render deploy/{systemd,env}/*.in for THIS checkout.
#
# systemd cannot do this itself: WorkingDirectory=, EnvironmentFile= and the
# ExecStart= binary path are all literal -- systemd expands ${VAR} only in
# ExecStart *arguments*, never in a path. So the units ship as templates with
# @SEC_SYS_ROOT_DIR@ / @EDGE_AI_ETC@ placeholders and are rendered here.
#
#   scripts/install_systemd.sh              # render + install the user units
#   scripts/install_systemd.sh --dry-run    # print what would be written
#
# Rendering is ALL this does. It never runs sudo and never talks to systemd:
# user units go to UNIT_DIR (your own directory), env files are rendered to
# deploy/rendered/ for you to install as root, and daemon-reload is yours to run.
# The remaining steps are printed at the end.
#
#   SEC_SYS_ROOT_DIR  repo root      (default: this checkout)
#   EDGE_AI_ETC       env-file dir   (default: /etc/edge-ai) -- baked into the units
#   UNIT_DIR          unit dir       (default: ~/.config/systemd/user)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EDGE_AI_ETC="${EDGE_AI_ETC:-/etc/edge-ai}"
UNIT_DIR="${UNIT_DIR:-$HOME/.config/systemd/user}"
STAGING="$REPO/deploy/rendered"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

render() {   # render <template> -> stdout
    sed -e "s|@SEC_SYS_ROOT_DIR@|$SEC_SYS_ROOT_DIR|g" \
        -e "s|@EDGE_AI_ETC@|$EDGE_AI_ETC|g" "$1"
}

echo "repo:      $SEC_SYS_ROOT_DIR"
echo "units   -> $UNIT_DIR"
echo "env     -> $STAGING  (copy to $EDGE_AI_ETC yourself)"
echo

# --- env files: rendered to staging only; never written to $EDGE_AI_ETC -------
for tpl in "$REPO"/deploy/env/*.env.in; do
    name="$(basename "$tpl" .in)"
    if [ "$DRY_RUN" = 1 ]; then
        echo "--- $STAGING/$name"; render "$tpl"; echo
        continue
    fi
    mkdir -p "$STAGING"
    render "$tpl" > "$STAGING/$name"
    echo "rendered $STAGING/$name"
done

# --- user units ---------------------------------------------------------------
for tpl in "$REPO"/deploy/systemd/*.service.in; do
    name="$(basename "$tpl" .in)"
    if [ "$DRY_RUN" = 1 ]; then
        echo "--- $UNIT_DIR/$name"; render "$tpl"; echo
        continue
    fi
    mkdir -p "$UNIT_DIR"
    render "$tpl" > "$UNIT_DIR/$name"
    echo "installed $UNIT_DIR/$name"
done

if [ "$DRY_RUN" = 1 ]; then
    echo "(dry run -- nothing written)"
    exit 0
fi

cat <<EOF

NEXT -- three steps, all yours to run:

  1. install the env files as root (units won't start without them)
       sudo install -d $EDGE_AI_ETC
       sudo install -m 644 $STAGING/*.env $EDGE_AI_ETC/

  2. pick up the new unit definitions
       systemctl --user daemon-reload

  3. start the fleet
       systemctl --user start edge-display     # then use http://localhost:8088/
     or in dependency order
       systemctl --user start edge-tracking edge-inference edge-ingest
EOF
