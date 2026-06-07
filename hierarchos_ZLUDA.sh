#!/bin/bash

# Check if ZLUDA is installed (assuming same structure or system install)
if [ -f "ZLUDA/bin/zluda" ]; then
    ZLUDA_BIN="ZLUDA/bin/zluda"
elif command -v zluda &> /dev/null; then
    ZLUDA_BIN="zluda"
else
    echo "❌ ZLUDA not found."
    echo "Please install ZLUDA or run setup.sh (if implemented for Linux ZLUDA)."
    exit 1
fi

echo "=================================================="
echo "== Running Hierarchos with ZLUDA (AMD Support) =="
echo "=================================================="
echo ""

# Run the modular CLI. We explicitly pass --device cuda because ZLUDA mocks CUDA.
"$ZLUDA_BIN" -- python hierarchos_cli.py "$@" --device cuda
