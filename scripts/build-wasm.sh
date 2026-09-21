#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Building Eigen-js WebAssembly ==="

# 1. Check if Emscripten environment is present
if ! command -v em++ >/dev/null 2>&1; then
  if [ -n "$EMSDK" ] && [ -f "$EMSDK/emsdk_env.sh" ]; then
    echo "Activating EMSDK from \$EMSDK ($EMSDK)..."
    # shellcheck source=/dev/null
    source "$EMSDK/emsdk_env.sh"
  else
    echo "Error: em++ not found in PATH."
    echo "Please activate your Emscripten environment (e.g. source /path/to/emsdk_env.sh) or set the EMSDK environment variable."
    exit 1
  fi
fi

echo "Using Emscripten: $(em++ -v 2>&1 | head -n 1)"

# 2. Check options
WITH_OSQP=true
for arg in "$@"; do
  case $arg in
    --no-osqp)
      WITH_OSQP=false
      shift
      ;;
    --with-osqp)
      WITH_OSQP=true
      shift
      ;;
  esac
done

mkdir -p "${ROOT_DIR}/build"

# 3. Build OSQP if requested
EXTRA_INCLUDES=()
EXTRA_LIBS=()
DEFINES=()

if [ "$WITH_OSQP" = true ]; then
  echo "--- Building OSQP submodule ---"
  if [ ! -d "${ROOT_DIR}/lib/osqp/.git" ] && [ ! -f "${ROOT_DIR}/lib/osqp/CMakeLists.txt" ]; then
    echo "Initializing lib/osqp submodule..."
    git -C "${ROOT_DIR}" submodule update --init --recursive lib/osqp
  fi

  cd "${ROOT_DIR}/lib/osqp"
  if [ ! -f "build/out/libosqpstatic.a" ]; then
    echo "Configuring OSQP with emcmake cmake..."
    emcmake cmake -S . -B build \
      -DOSQP_BUILD_SHARED_LIB=OFF \
      -DOSQP_BUILD_STATIC_LIB=ON \
      -DOSQP_BUILD_DEMO_EXE=OFF \
      -DOSQP_ENABLE_INTERRUPT=OFF
    emmake make -C build -j4
  fi
  cd "${ROOT_DIR}"

  EXTRA_INCLUDES+=("-I" "lib/osqp/include/public" "-I" "lib/osqp/build/include/public")
  EXTRA_LIBS+=("lib/osqp/build/out/libosqpstatic.a")
else
  echo "--- Building WITHOUT OSQP ---"
  DEFINES+=("-D" "NO_OSQP")
fi

# 4. Check lib/eigen
if [ ! -f "${ROOT_DIR}/lib/eigen/Eigen/Dense" ]; then
  echo "Initializing lib/eigen submodule on branch master..."
  git -C "${ROOT_DIR}" submodule update --init lib/eigen
fi

# 5. Compile WebAssembly binary
echo "--- Compiling embind.cc to WebAssembly ---"
cd "${ROOT_DIR}"

em++ \
  "${DEFINES[@]}" \
  -I lib/eigen \
  -I src \
  "${EXTRA_INCLUDES[@]}" \
  "${EXTRA_LIBS[@]}" \
  -s DISABLE_EXCEPTION_CATCHING=0 \
  -s ASSERTIONS=0 \
  -O3 \
  -s STACK_SIZE=8388608 \
  -s INITIAL_MEMORY=67108864 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s INCOMING_MODULE_JS_API=wasmBinary,locateFile,onRuntimeInitialized,instantiateWasm,print,printErr \
  --bind \
  -o build/eigen_gen.js \
  src/cpp/embind.cc

echo "=== Build completed successfully! ==="
echo "Artifacts generated in ${ROOT_DIR}/build:"
ls -lh "${ROOT_DIR}/build/eigen_gen.js" "${ROOT_DIR}/build/eigen_gen.wasm"
