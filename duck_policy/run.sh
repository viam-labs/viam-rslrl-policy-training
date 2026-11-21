#!/usr/bin/env bash
# Module entrypoint, used as a local module: run.sh lives in duck_policy/ and PYTHONPATH points at the repo root, so the package imports from this checkout. The venv stays in duck_policy/.
# The env is built with uv: `uv venv` needs no ensurepip, which images such as the Viam sim sandbox
# lack (no python3-venv). uv comes from PATH, or from its GitHub release into ./bin on first run.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ requirements.txt -nt .venv/.installed ]; then
  UV=$(command -v uv || true)
  if [ -z "$UV" ]; then
    case "$(uname -s)" in Linux) os=unknown-linux-gnu ;; Darwin) os=apple-darwin ;; *) echo "unsupported OS" >&2; exit 1 ;; esac
    triple="$(uname -m | sed 's/arm64/aarch64/')-$os"
    mkdir -p bin
    python3 - "$triple" <<'PY'
import io, sys, tarfile, urllib.request
url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{sys.argv[1]}.tar.gz"
with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())) as t:
    m = next(m for m in t.getmembers() if m.name.endswith("/uv"))
    m.name = "uv"
    t.extract(m, "bin")
PY
    UV="$PWD/bin/uv"
  fi
  rm -rf .venv
  "$UV" venv -q --python 3.12 .venv
  "$UV" pip install -q --python .venv/bin/python -r requirements.txt
  touch .venv/.installed
fi
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m duck_policy.main "$@"
