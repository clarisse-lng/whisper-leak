#!/usr/bin/env bash
# Run this once on the host to pre-download all pip wheels.
# The Dockerfile then installs from this local cache — no internet needed per build.
set -euo pipefail

mkdir -p pip_packages
pip download -r requirements.txt -d pip_packages
echo "Packages downloaded to ./pip_packages/"
