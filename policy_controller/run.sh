#!/usr/bin/env bash
# Module entrypoint. The tarball root holds policy_controller/ (the package), run.sh and requirements.txt.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ requirements.txt -nt .venv/.installed ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
  touch .venv/.installed
fi
exec .venv/bin/python -m policy_controller.main "$@"
