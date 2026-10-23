#!/usr/bin/env bash
# gles_run with a hard per-dispatch timeout, for hardware runs. Pass this script
# as GLES_RUN to the tests. A timeout is a stop condition for the gate, never a
# reason to retry. MESA_WRAPPER (optional) is a launcher such as the isolated
# run-with-mesa24.sh that takes the command to run as its arguments.
# usage: GLES_RUN_BIN=/path/to/gles_run [GLES_TIMEOUT=60] [MESA_WRAPPER=...] gles_run_guarded.sh ARGS...
exec timeout -k 5 "${GLES_TIMEOUT:-60}" ${MESA_WRAPPER:+"$MESA_WRAPPER"} "${GLES_RUN_BIN:?set GLES_RUN_BIN}" "$@"
