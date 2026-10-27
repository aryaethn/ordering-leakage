#!/bin/bash
# extract_positions.sh <YYYYMMDD>
# Positions-only extraction for the swap-adversary rerun (12-phase3-results.md, section H).
# Reads the Blockchair transaction dump straight from the .tsv.gz (or .tsv if already
# unzipped) and writes data/derived/blocks_<day>.csv, using the base-fee table that
# extract_day.sh already produced. No mempool file needed. Afterwards the dump can be deleted.
set -euo pipefail
D8=$1
ROOT="$(cd "$(dirname "$0")/.." && pwd)/data"
OUT="$ROOT/derived"
SRC="$ROOT/txs/blockchair_ethereum_transactions_${D8}.tsv.gz"
if [ -f "$SRC" ]; then CAT="gzip -dc"; else SRC="${SRC%.gz}"; CAT="cat"; fi
[ -f "$SRC" ] || { echo "missing $SRC(.gz)"; exit 1; }
$CAT "$SRC" | awk -F'\t' -v OFS=',' '
  NR==FNR { bf[$1]=$2; next }
  FNR==1  { print "block_number","tx_hash","position","sender","priority_fee","base_fee","gas_used","block_gas_limit"; next }
  ($6=="call" || $6=="call_tree") && ($1 in bf) {
      pf = $18 - bf[$1]; if (pf < 0) pf = 0;
      print $1, $3, $2, $7, pf, bf[$1], $16, 0
  }
' "$OUT/basefee_${D8}.tsv" - > "$OUT/blocks_${D8}.csv"
echo "done ${D8}: $(wc -l < "$OUT/blocks_${D8}.csv") rows"
