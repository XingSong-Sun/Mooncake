#!/usr/bin/env bash
set -euo pipefail

VLLM_PYTHON="${VLLM_PYTHON:-$(python3 -c 'import sys; print(sys.executable)')}"
echo "[INFO] 使用 Python 解释器: ${VLLM_PYTHON}"
"${VLLM_PYTHON}" -c 'import sys; print(f"[INFO] Python 版本: {sys.version.split()[0]}")'


##############################################
#                  参数解析                  #
##############################################

SCRIPT_NAME="${0##*/}"
usage() {
    cat >&2 <<EOF
用法: ${SCRIPT_NAME} <npu|gpu>

  npu  编译 Ascend NPU 版本（USE_ASCEND_HETEROGENEOUS=ON）
  gpu  编译 NVIDIA GPU 版本（USE_CUDA=ON）
EOF
}

BACKEND="${1:-}"
case "${BACKEND}" in
    npu)
        MOONCAKE_CMAKE_FLAG="-DUSE_ASCEND_HETEROGENEOUS=ON"
        ;;
    gpu)
        MOONCAKE_CMAKE_FLAG="-DUSE_CUDA=ON"
        ;;
    *)
        usage
        exit 2
        ;;
esac


##############################################
#                    GPU 分支                #
##############################################

if [[ "${BACKEND}" == "gpu" ]]; then
    # nvidia_cutlass_dsl 的 .pth 会把 dsl_packages 插到 sys.path 最前，
    # 导致 mooncake 的 CMake 把 Python 包装进错误目录。
    # 配置期间临时停用它，脚本退出时自动恢复。
    CUTLASS_PTH="$("${VLLM_PYTHON}" - <<'PY'
import os, site
for p in site.getsitepackages():
    candidate = os.path.join(p, "nvidia_cutlass_dsl_packages.pth")
    if os.path.exists(candidate):
        print(candidate)
        break
PY
    )"

    CUTLASS_PTH_DISABLED=""
    if [[ -n "${CUTLASS_PTH}" && -f "${CUTLASS_PTH}" ]]; then
        CUTLASS_PTH_DISABLED="${CUTLASS_PTH}.disabled"
        mv "${CUTLASS_PTH}" "${CUTLASS_PTH_DISABLED}"
        echo "[INFO] 已临时停用 ${CUTLASS_PTH}"
    elif [[ -n "${CUTLASS_PTH}" && -f "${CUTLASS_PTH}.disabled" ]]; then
        # 上次运行异常中断时可能留下 .disabled，本次直接复用
        CUTLASS_PTH_DISABLED="${CUTLASS_PTH}.disabled"
        echo "[INFO] 复用上次遗留的临时停用状态: ${CUTLASS_PTH_DISABLED}"
    fi

    restore_cutlass_pth() {
        if [[ -n "${CUTLASS_PTH_DISABLED}" && -f "${CUTLASS_PTH_DISABLED}" ]]; then
            mv "${CUTLASS_PTH_DISABLED}" "${CUTLASS_PTH}"
            echo "[INFO] 已恢复 ${CUTLASS_PTH}"
        fi
    }
    trap restore_cutlass_pth EXIT
    rm -rf build
    mkdir build
    cd build
    cmake .. \
            "${MOONCAKE_CMAKE_FLAG}" \
            -DPython3_EXECUTABLE="${VLLM_PYTHON}" \
            -DPython_EXECUTABLE="${VLLM_PYTHON}" \
            -DPYTHON_EXECUTABLE="${VLLM_PYTHON}"
    make -j"$(nproc)"

    make install

    cd ..

    # make install 已把编译产物装进该解释器的 site-packages，无需任何 PYTHONPATH。
    "${VLLM_PYTHON}" - <<'PY'
import mooncake
import mooncake.engine
import mooncake.store

print(f"[INFO] mooncake 安装位置: {mooncake.__file__}")
print(f"[INFO] engine 安装位置: {mooncake.engine.__file__}")
print(f"[INFO] store 安装位置: {mooncake.store.__file__}")
PY
fi


##############################################
#                    NPU 分支                #
##############################################

if [[ "${BACKEND}" == "npu" ]]; then
    rm -rf build
    mkdir build
    cd build
    cmake .. \
            "${MOONCAKE_CMAKE_FLAG}" \
            -DPython3_EXECUTABLE="${VLLM_PYTHON}" \
            -DPython_EXECUTABLE="${VLLM_PYTHON}" \
            -DPYTHON_EXECUTABLE="${VLLM_PYTHON}"
    make -j"$(nproc)"

    make install

    # 验证目标解释器能否加载刚编译的 mooncake 包（在 /tmp 下执行，避免误用仓库内的源码目录）
    INSTALLED_PACKAGE="$(
        cd /tmp &&
        "${VLLM_PYTHON}" -c 'import mooncake; print(mooncake.__file__)' || true
    )"
    if [ -z "${INSTALLED_PACKAGE}" ]; then
        echo "[ERROR] 编译完成，但 ${VLLM_PYTHON} 无法导入 mooncake 包" >&2
        exit 1
    fi

    MOONCAKE_SITE="$(dirname "$(dirname "${INSTALLED_PACKAGE}")")"
    echo "[INFO] mooncake 已安装到: ${MOONCAKE_SITE}"
    echo "[INFO] ${VLLM_PYTHON} 无需设置 PYTHONPATH，会直接加载上面的编译产物。"
    echo "[INFO] 如需用其它解释器加载（须与编译时同版本），请执行:"
    echo "[INFO]   export PYTHONPATH=${MOONCAKE_SITE}:\$PYTHONPATH"

    if ! (cd /tmp && "${VLLM_PYTHON}" -c 'import mooncake.engine, mooncake.store'); then
        echo "[WARN] mooncake.engine / mooncake.store 导入失败，请检查运行时依赖" >&2
    fi
fi
