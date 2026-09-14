#!/bin/bash
# extract_day.sh <YYYYMMDD> <YYYY-MM-DD>
# Reduces one day of Blockchair dumps + one day of mempool-dumpster CSV to the
# compact schema ordering_leakage.py expects. Streams; memory stays flat.
set -euo pipefail
D8=$1; DASH=$2
ROOT="$(cd "$(dirname "$0")/.." && pwd)/data"
OUT="$ROOT/derived"
mkdir -p "$OUT"

# 1. blocks meta + base fee table.
#    base_fee = (fee_total - reward) / gas_used   [validated against min tx gas price]
python3 - "$ROOT/blocks/blockchair_ethereum_blocks_${D8}.tsv" "$OUT" "$D8" <<'PY'
import csv, sys, datetime
src, out, d8 = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f, \
     open(f"{out}/blocks_meta_{d8}.csv", "w", newline="") as mo, \
     open(f"{out}/basefee_{d8}.tsv", "w") as bo:
    r = csv.DictReader(f, delimiter="\t")
    w = csv.writer(mo)
    w.writerow(["block_number", "timestamp_ms", "base_fee", "gas_used", "gas_limit"])
    for row in r:
        gu = float(row["gas_used"])
        if gu <= 0:
            continue
        bf = (float(row["fee_total"]) - float(row["reward"])) / gu
        ts = int(datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                 .replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        w.writerow([row["id"], ts, int(bf), row["gas_used"], row["gas_limit"]])
        bo.write(f"{row['id']}\t{int(bf)}\n")
PY

# 2. per-transaction rows. Only real top-level transactions (call, call_tree);
#    synthetic_coinbase is the block reward row and is dropped.
awk -F'\t' -v OFS=',' '
  NR==FNR { bf[$1]=$2; next }
  FNR==1  { print "block_number","tx_hash","position","sender","priority_fee","base_fee","gas_used","block_gas_limit"; next }
  ($6=="call" || $6=="call_tree") && ($1 in bf) {
      pf = $18 - bf[$1];
      if (pf < 0) pf = 0;
      print $1, $3, $2, $7, pf, bf[$1], $16, 0
  }
' "$OUT/basefee_${D8}.tsv" "$ROOT/txs/blockchair_ethereum_transactions_${D8}.tsv" > "$OUT/blocks_${D8}.csv"

# 3. mempool, reduced to the four columns the measurement needs.
awk -F',' -v OFS=',' '
  FNR==1 { print "tx_hash","first_seen_ms","sender","priority_fee","fee_cap","included_at"; next }
  { print $2, $1, $4, $10, $11, $15 }
' "$ROOT/mempool/${DASH}.csv" > "$OUT/mempool_${D8}.csv"

echo "done ${D8}:"
wc -l "$OUT/blocks_meta_${D8}.csv" "$OUT/blocks_${D8}.csv" "$OUT/mempool_${D8}.csv"
