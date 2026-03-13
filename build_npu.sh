#!/usr/bin/env bash
set -euo pipefail

VLLM_PYTHON="${VLLM_PYTHON:-$(python3 -c 'import sys; print(sys.executable)')}"
echo "[INFO] 使用 Python 解释器: ${VLLM_PYTHON}"
"${VLLM_PYTHON}" -c 'import sys; print(f"[INFO] Python 版本: {sys.version.split()[0]}")'

rm -rf build
mkdir build
cd build
cmake .. \
	-DUSE_ASCEND_HETEROGENEOUS=ON \
	-DPython3_EXECUTABLE="${VLLM_PYTHON}" \
	-DPython_EXECUTABLE="${VLLM_PYTHON}" \
	-DPYTHON_EXECUTABLE="${VLLM_PYTHON}"
make -j"$(nproc)"