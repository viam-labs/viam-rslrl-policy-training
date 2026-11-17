#!/usr/bin/env bash
# Builds dist/rl-insert-train.tar.gz, the training-script package Viam's launcher pip-installs.
# Staged in its own directory: the repo's pyproject.toml describes the dev env, and pip/build
# would package that instead of setup.py.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf dist && mkdir -p dist/stage
cp -R setup.py MANIFEST.in model rl_insert dist/stage/
# COPYFILE_DISABLE: macOS tar would otherwise add AppleDouble ._* entries.
(cd dist/stage && COPYFILE_DISABLE=1 uvx --from build pyproject-build --sdist --outdir .. . >/dev/null)
mv dist/rl_insert_train-*.tar.gz dist/rl-insert-train.tar.gz
rm -rf dist/stage
tar -tzf dist/rl-insert-train.tar.gz | grep -cE "\.py$|\.stl$|\.urdf$" | xargs echo "files (py/stl/urdf):"
