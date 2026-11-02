#!/usr/bin/env bash
# ==============================================================================
# CLEANUP SCRIPT: Restore environment and remove project artifacts
# ==============================================================================
set -euo pipefail

PROJECT_DIR="/home/neymar-jr/DSC Project"
TRACKING_FILE="${PROJECT_DIR}/installed_libraries.txt"

echo "=== Antigravity Project Cleanup Initiated ==="
echo "Project Directory: ${PROJECT_DIR}"

# 1. Parse and uninstall any additional libraries tracked
if [ -f "${TRACKING_FILE}" ]; then
    echo "Checking for additional libraries to uninstall..."
    # Extract package names from the additional section (lines after the header)
    ADDITIONAL_PKGS=$(awk '/ADDITIONAL \/ NEWLY INSTALLED LIBRARIES/{flag=1; next} flag && /^[a-zA-Z0-9_-]+==/{print $1}' "${TRACKING_FILE}" | cut -d'=' -f1)
    if [ -n "${ADDITIONAL_PKGS}" ]; then
        echo "Found packages to remove:"
        echo "${ADDITIONAL_PKGS}"
        for pkg in ${ADDITIONAL_PKGS}; do
            echo "Uninstalling ${pkg}..."
            pip uninstall -y "${pkg}" || true
        done
    else
        echo "No additional packages recorded to uninstall."
    fi
fi

# 2. Prompt or clean temporary outputs / caches / build files
echo "Cleaning Python cache files and build artifacts..."
find "${PROJECT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${PROJECT_DIR}" -type f -name "*.pyo" -delete 2>/dev/null || true
find "${PROJECT_DIR}" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

echo "Purging pip cache and temporary runtime directories..."
pip cache purge 2>/dev/null || true
rm -rf /tmp/torchinductor_* 2>/dev/null || true

echo "=== Cleanup script completed successfully. ==="
