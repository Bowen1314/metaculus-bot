#!/bin/sh
# One bot pass in Docker. Used by the systemd timer; safe to run by hand.
#   deploy/run-once.sh                 -> run --live (does nothing unless armed = true)
#   deploy/run-once.sh doctor          -> setup check
#   deploy/run-once.sh dry-run         -> forecasts on the unscored testing area, nothing submitted
set -eu
DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$DIR/state" "$DIR/logs"
[ $# -gt 0 ] || set -- run --live
exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --memory 700m --cpus 1 \
  --log-driver json-file --log-opt max-size=5m --log-opt max-file=2 \
  -v "$DIR/.env:/app/.env:ro" \
  -v "$DIR/config.toml:/app/config.toml:ro" \
  -v "$DIR/state:/app/state" \
  -v "$DIR/logs:/app/logs" \
  metaculus-bot "$@"
