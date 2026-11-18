#!/usr/bin/env bash
# Builds dist/rl-policy-module.tar.gz for `viam module upload` (run it from policy_controller/, where meta.json is).
set -euo pipefail
cd "$(dirname "$0")"
stage=$(mktemp -d)
mkdir "$stage/policy_controller"
cp policy_controller/*.py "$stage/policy_controller/"
cp policy_controller/run.sh policy_controller/requirements.txt policy_controller/meta.json "$stage/"
mkdir -p dist
COPYFILE_DISABLE=1 tar -czf dist/rl-policy-module.tar.gz -C "$stage" .
rm -rf "$stage"
tar -tzf dist/rl-policy-module.tar.gz
