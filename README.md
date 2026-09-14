# ordering-leakage

Measurement code and results for the WPPT 2026 abstract *What the Realized Order Reveals: A Leakage
Bound for Randomness-Fair, Censorship-Evident Transaction Ordering*.

The paper defines a security game in which the adversary's view is cut at the moment the realized
transaction order is fixed and before any payload opens, proves a leakage lower bound for any
mechanism whose inclusion decision responds to the fee, and measures both leakage channels on
Ethereum mainnet. This repository is the measurement half.

**Everything here is observational.** No proposed ordering rule is simulated. The only synthetic
object is a null control, a content-independent permutation of each block's included set, which
exists so the estimators can be shown to read zero when the order carries no fee information.

## Headline results

Two three-day windows of mainnet, about 8.9M transactions in 42,793 blocks.

| | 2024-03-19/21 | 2025-11-04/06 | null control |
|---|---|---|---|
| order identifies the higher payer | **0.7915** [0.7813, 0.8021] | **0.6118** [0.6043, 0.6195] | 0.0004 / 0.0023 |
| inclusion responsiveness gap, base-fee normalizer | 0.145 [0.134, 0.169] | 0.092 [0.081, 0.109] | |
| inclusion responsiveness gap, median-tip normalizer | 0.352 [0.324, 0.381] | 0.120 [0.103, 0.142] | |
| explicit swap adversary, chosen pair | 0.772 | 0.529 | 0.770 / 0.527 |

All figures are advantages, that is, |2 * accuracy - 1|, and all are **lower bounds**: the adversary
is explicit and suboptimal. Intervals are a block bootstrap with 400 replicates, blocks being the
sampling unit. Row 1 is computed over 58.6M and 79.5M admissible transaction pairs.

Two independent channels, each near 0.8. For two transactions that both landed, the realized order
identifies which sender paid more about nine times in ten in the earlier window. For an adversary
free to choose the two fee levels, inclusion alone carries 0.77 and the order is redundant.

## Reproducing

Data is not in this repository, roughly 13 GB. Fetch it yourself:

- Public mempool: the Flashbots [mempool dumpster](https://mempool-dumpster.flashbots.net/) daily
  transaction-metadata CSV, `ethereum/mainnet/<YYYY-MM>/<YYYY-MM-DD>.csv.zip`, about 100 MB per day
  zipped. Not the parquet.
- Blocks and transactions: [Blockchair](https://gz.blockchair.com/ethereum/) daily dumps,
  `blockchair_ethereum_blocks_<YYYYMMDD>.tsv` and `blockchair_ethereum_transactions_<YYYYMMDD>.tsv`.

Lay them out as `data/mempool/<YYYY-MM-DD>.csv`, `data/blocks/...tsv`, `data/txs/...tsv`, then:

```bash
./code/extract_day.sh 20240319 2024-03-19          # once per day, about 60s each
python3 code/run_window.py --name congested --days 20240319 20240320 20240321 --step position
python3 code/run_window.py --name congested --days 20240319 20240320 20240321 --step candidates
python3 code/run_window.py --name congested --days 20240319 20240320 20240321 --step metrics
python3 code/finalize.py   --name congested --days 20240319 20240320 20240321
```

Needs Python 3.10+, pandas and numpy. `python3 code/ordering_leakage.py --selftest` runs the
estimators against three synthetic mechanisms with known ground truth and should print
SELF-TEST PASSED.

## Two things worth knowing before you trust a number

**Base fee is reconstructed, not read.** Blockchair's dumps do not carry a base fee column, contrary
to their API documentation. It is recovered per block as
`(fee_total - reward) / gas_used` and validated: across 50 consecutive blocks no block had a minimum
included gas price below the derived base fee, and 48 of 50 matched it to within 0.02 gwei, which is
what a block containing a zero-tip transaction should look like.

**The inclusion estimator is confounded if you take the obvious route.** Asking, for every block a
transaction was pending in, whether it was included there conditions on survival, and that biases
high-fee bins down hardest: a transaction paying sixteen times the going rate that is still pending
after two slots is one that cannot be included for a reason unrelated to price. Conditioning on age
makes it worse, not better (0.83, 0.35, 0.18 for the top bin at ages 0, at least 1, at least 2
slots). The reported numbers use each transaction's first candidacy under one slot old.

## Files

- `code/extract_day.sh` reduces one day of raw dumps to a compact schema.
- `code/ordering_leakage.py` estimators and the synthetic self-test.
- `code/run_window.py` per-window pipeline in three steps.
- `code/finalize.py` block bootstrap, normalizer sensitivity, survivorship diagnostics.
- `results/` the outputs behind the table above.

## License

MIT, see LICENSE.
