#!/usr/bin/env bash
# Module entrypoint. The tarball root holds policy_controller/ (the package), run.sh and requirements.txt.
# Images such as the Viam sim sandbox ship python3 without ensurepip (no python3-venv), so the venv is
# made without pip and filled with pip's official zipapp.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ requirements.txt -nt .venv/.installed ]; then
  rm -rf .venv
  python3 -m venv --without-pip .venv
  [ -f pip.pyz ] || python3 -c "import urllib.request as u; u.urlretrieve('https://bootstrap.pypa.io/pip/pip.pyz', 'pip.pyz')"
  .venv/bin/python pip.pyz install -q --disable-pip-version-check -r requirements.txt
  touch .venv/.installed
fi
exec .venv/bin/python -m policy_controller.main "$@"
