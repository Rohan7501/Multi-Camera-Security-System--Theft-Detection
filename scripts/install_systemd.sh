#!/usr/bin/env bash
# Render deploy/{systemd,env}/*.in for THIS checkout into deploy/rendered/.
#
# systemd cannot do this itself: WorkingDirectory=, EnvironmentFile= and the
# ExecStart= binary path are all literal -- systemd expands ${VAR} only in
# ExecStart *arguments*, never in a path. So the units ship as templates with
# @SEC_SYS_ROOT_DIR@ / @EDGE_AI_ETC@ placeholders and are rendered here.
#
#   scripts/install_systemd.sh              # render into deploy/rendered/
#   scripts/install_systemd.sh --dry-run    # print what would be rendered
#
# Rendering is ALL this does. Nothing is installed, no sudo is run, and systemd
# is never contacted -- you copy the files and reload yourself. The commands to
# do that are printed at the end and documented in README.md.
#
#   SEC_SYS_ROOT_DIR  repo root      (default: this checkout)
#   EDGE_AI_ETC       env-file dir   (default: /etc/edge-ai) -- baked into the units
#   UNIT_DIR          unit dir       (default: ~/.config/systemd/user) -- printed only
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
            sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

render() {   # render <template> -> stdout
    sed -e "s|@SEC_SYS_ROOT_DIR@|$SEC_SYS_ROOT_DIR|g" \
        -e "s|@EDGE_AI_ETC@|$EDGE_AI_ETC|g" "$1"
}

echo "repo:     $SEC_SYS_ROOT_DIR"
echo "rendered: $STAGING"
echo

# Everything lands in staging. The two kinds are told apart by extension later,
# which is what makes the printed copy commands a pair of clean globs.
for tpl in "$REPO"/deploy/env/*.env.in "$REPO"/deploy/systemd/*.service.in; do
    name="$(basename "$tpl" .in)"
    if [ "$DRY_RUN" = 1 ]; then
        echo "--- $STAGING/$name"; render "$tpl"; echo
        continue
    fi
    mkdir -p "$STAGING"
    render "$tpl" > "$STAGING/$name"
    echo "rendered $STAGING/$name"
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

  2. install the user units, then pick them up
       install -d $UNIT_DIR
       install -m 644 $STAGING/*.service $UNIT_DIR/
       systemctl --user daemon-reload

  3. start the fleet
       systemctl --user start edge-display     # then use http://localhost:8088/
     or in dependency order
       systemctl --user start edge-tracking edge-inference edge-ingest
EOF
