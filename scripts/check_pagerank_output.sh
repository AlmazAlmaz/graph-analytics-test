#!/usr/bin/env bash
set -euo pipefail

file="$1"

test -s "$file"
head -n 1 "$file" | grep -q '^vertex,rank$'
awk -F, '
  NR > 1 {
    rows += 1
    sum += $2
    if ($2 < 0) {
      bad = 1
    }
  }
  END {
    if (rows == 0 || bad || sum < 0.999999 || sum > 1.000001) {
      exit 1
    }
  }
' "$file"
