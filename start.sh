#!/usr/bin/env bash
set -euo pipefail

# Render sets PORT; default to 7860 if not present
PORT="${PORT:-7860}"

# Use either a remote URL or a local file path:
BOOT_URL="${PRODUCT_CSV_URL:-}"   # e.g. https://raw.githubusercontent.com/.../cleaned_products.csv
BOOT_FILE="${PRODUCT_CSV:-}"      # e.g. /app/data/product.csv

# Background job: wait for /health, then POST /data/load once
(
  echo "[boot] Waiting for app on /health..."
  for i in {1..60}; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "[boot] App healthy, loading CSV once..."
      if [[ -n "$BOOT_URL" ]]; then
        tmp="/tmp/boot.csv"
        if curl -fsSL "$BOOT_URL" -o "$tmp"; then
          curl -fsS -X POST -F "file=@${tmp}" "http://127.0.0.1:${PORT}/data/load" \
            && echo "[boot] Loaded CSV from URL" && exit 0
        fi
        echo "[boot] Failed to load from URL; skipping."
      elif [[ -n "$BOOT_FILE" && -f "$BOOT_FILE" ]]; then
        curl -fsS -X POST -F "file=@${BOOT_FILE}" "http://127.0.0.1:${PORT}/data/load" \
          && echo "[boot] Loaded CSV from file" && exit 0
      else
        echo "[boot] No BOOT_URL or BOOT_FILE set; nothing to load."
        exit 0
      fi
    fi
    sleep 1
  done
  echo "[boot] Timeout waiting for /health; skipping autoload."
) &

# Hand off to whatever CMD/render dockerCommand specifies (Gunicorn by default)
exec "$@"
