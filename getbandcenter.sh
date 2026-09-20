#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <last_input>"
  echo "Example: $0 1"
  exit 1
fi

LAST_INPUT="$1"

{
  echo "503"
  echo "1"
  echo "Y"
  echo "-15 15"
  echo "1"
  echo "$LAST_INPUT"
} | vaspkit
