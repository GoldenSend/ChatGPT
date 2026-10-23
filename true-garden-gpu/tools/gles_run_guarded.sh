#!/usr/bin/env bash
# gles_run with a hard per-dispatch timeout, for hardware runs. Pass this script
# as GLES_RUN to the tests. A timeout is a stop condition for the gate, never a
# reason to retry.
# usage: GLES_RUN_BIN=/path/to/gles_run [GLES_TIMEOUT=60] gles_run_guarded.sh ARGS...
exec timeout -k 5 "${GLES_TIMEOUT:-60}" "${GLES_RUN_BIN:?set GLES_RUN_BIN}" "$@"
