#!/bin/bash

# Run all Ragas example scripts and report per-file status.
# This script can be invoked from any directory.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ ! -d "${EXAMPLES_DIR}" ]; then
  echo "Examples directory not found: ${EXAMPLES_DIR}" >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Note: OPENAI_API_KEY is not set. LLM-dependent examples may skip." >&2
fi

mapfile -t FILES < <(find "${EXAMPLES_DIR}" -maxdepth 1 -type f -name "*.py" | sort)

total=0
ok=0
skipped=0
failed=0

echo "Running ${#FILES[@]} example scripts...";
echo

for file in "${FILES[@]}"; do
  total=$((total + 1))
  rel_path="${file#${PROJECT_DIR}/}"
  echo "==> ${rel_path}"
  # Capture output while preserving exit code
  out=""
  if ! out="$(uv run --project "${PROJECT_DIR}" "${file}" 2>&1)"; then
    failed=$((failed + 1))
    echo "STATUS: FAILED"
    echo "OUTPUT:"; echo "${out}" | sed 's/^/  /'
    echo
    continue
  fi

  # Detect graceful skips in output
  if echo "${out}" | grep -qiE "Skipping LLM example|Skipping a sample"; then
    skipped=$((skipped + 1))
    echo "STATUS: SKIPPED"
    echo "OUTPUT:"; echo "${out}" | sed 's/^/  /'
    echo
    continue
  fi

  ok=$((ok + 1))
  echo "STATUS: SUCCESS"
  echo "OUTPUT:"; echo "${out}" | sed 's/^/  /'
  echo
done

echo "Summary: total=${total} success=${ok} skipped=${skipped} failed=${failed}"

if [ "${failed}" -gt 0 ]; then
  exit 1
fi
exit 0
