#!/usr/bin/env bash
# Builds dist/microduck-train.tar.gz for `viam training-script upload`. Staged on its own: the repo's
# model/training.py is the peg-insert entry, so the stage gets a model/training.py shim to duck_train.
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
stage=$(mktemp -d)
mkdir -p "$stage/model" "$stage/duck_train" dist
cp model/__init__.py model/harness.py model/viam_upload.py "$stage/model/"
cat > "$stage/model/training.py" <<'PY'
"""Viam runs `python3 -m model.training`: the MicroDuck job's entry is duck_train.train.main."""
from duck_train.train import main

if __name__ == "__main__":
    main()
PY
cp duck_train/__init__.py duck_train/train.py duck_train/package.py duck_train/robot.py duck_train/dance.py duck_train/dance_rewards.py duck_train/dance_task.py "$stage/duck_train/"
cp duck_train/setup.py "$stage/setup.py"
(cd "$stage" && COPYFILE_DISABLE=1 uvx --from build pyproject-build --sdist --outdir "$root/dist" . >/dev/null)
mv dist/microduck_train-*.tar.gz dist/microduck-train.tar.gz
rm -rf "$stage"
tar -tzf dist/microduck-train.tar.gz
