#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/.streamlit

# Escape for TOML double-quoted strings
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# Required env vars (keep names consistent with Cloud Run mappings)
: "${PASSWORD:?Missing PASSWORD env var}"
: "${GORILA_API_KEY:?Missing GORILA_API_KEY env var}"
: "${INT_API_KEY:?Missing INT_API_KEY env var}"

cat >/app/.streamlit/secrets.toml <<EOF
[auth]
PASSWORD = "$(esc "$PASSWORD")"

[apis]
GORILA_API_KEY = "$(esc "$GORILA_API_KEY")"
INT_API_KEY = "$(esc "$INT_API_KEY")"
EOF

chmod 600 /app/.streamlit/secrets.toml

exec streamlit run relatorio_comissoes.py --server.port=8080 --server.address=0.0.0.0