#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/.streamlit

# Escape content for TOML strings
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# Validate required env vars
: "${PASSWORD:?Missing PASSWORD env var}"
: "${GORILA_KEY:?Missing GORILA_API_KEY env var}"


cat >/app/.streamlit/secrets.toml <<EOF
[auth]
PASSWORD = "$(esc "$PASSWORD")"

[apis]
GORILA_API_KEY = "$(esc "$GORILA_KEY")"

EOF

# Optional hardening:
chmod 600 /app/.streamlit/secrets.toml

exec streamlit run relatorio_comissoes.py --server.port=8080 --server.address=0.0.0.0