#!/usr/bin/env bash
# Compatibility shim — delegates to run.sh golden
exec "$(dirname "$0")/run.sh" golden "$@"
