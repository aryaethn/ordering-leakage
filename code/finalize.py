"""
finalize.py

Closes the three open items left by the first measurement pass.

1. Error bars. Both estimators are decomposed into per-block sufficient statistics
   once, then the blocks are resampled with replacement. Bootstrapping the block
   sample is the right unit here: blocks are the sampling unit, transactions inside
   a block are not independent of each other.

2. Normalizer sensitivity. Delta_p is recomputed with the block's base fee as the
   normalizer instead of the median included tip, to test whether the U shape in
   congestion is an artifact of a normalizer that itself moves with congestion.

3. The declining top tail. Diagnostics on what sits in the high-fee bins: how many
   of those candidates are never included anywhere in the feed, and how many times
   the same transaction is counted. A per-transaction estimator that takes each
   transaction's FIRST candidacy only is reported alongside the per-row one, which
   removes the length bias that favours transactions that linger.

Usage:
    python3 code/finalize.py --name congested --days 20240319 20240320 20240321
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordering_leakage import apply_random_control, draw_R  # noqa: E402
from run_window import DERIVED, load_window  # noqa: E402

BIN_EDGES = [-np.inf, -4, -2, -1, -0.25, 0.25, 1, 2, 4, np.inf]
BIN_LABELS = ["<-4", "[-4,-2)", "[-2,-1)", "[-1,-.25)", "[-.25,.25)",
              "[.25,1)", "[1,2)", "[2,4)", ">=4"]


# --------------------------------------------------------------------------
# 1. position advantage with a block bootstrap
# --------------------------------------------------------------------------

def per_block_pair_stats(blocks: pd.DataFrame, position_col: str) -> pd.DataFrame:
    rows = []
    for b, g in blocks.groupby("block_number", sort=False):
        pos = g[position_col].to_numpy()
        fee = g["priority_fee"].to_numpy()
        snd = g["sender"].to_numpy()
        k = len(pos)
        if k < 2:
            continue
        i, j = np.triu_indices(k, k=1)
        ok = (snd[i] != snd[j]) & (fee[i] != fee[j])
        i, j = i[ok], j[ok]
        if len(i) == 0:
            continue
        conc = (fee[i] > fee[j]) == (pos[i] < pos[j])
        rows.append((b, len(i), int(np.count_nonzero(conc))))
    return pd.DataFrame(rows, columns=["block_number", "n_pairs", "n_conc"])


def bootstrap_ratio(n: np.ndarray, k: np.ndarray, reps: int,
                    rng: np.random.Generator) -> tuple[float, float, float]:
    """Point estimate and percentile CI for advantage = |2*k/n - 1|."""
    point = abs(2 * k.sum() / n.sum() - 1)
    m = len(n)
    draws = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, m, size=m)
        draws[r] = abs(2 * k[idx].sum() / n[idx].sum() - 1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------
# 2 and 3. inclusion curve, two normalizers, two estimators, with bootstrap
# --------------------------------------------------------------------------

def binned_counts(cands: pd.DataFrame, norm: pd.Series) -> pd.DataFrame:
    c = cands.join(norm.rename("norm"), on="block_number")
    c = c[(c["norm"] > 0) & (c["priority_fee"] > 0)]
    c = c.assign(bin=pd.cut(np.log2(c["priority_fee"] / c["norm"]),
                            bins=BIN_EDGES, labels=BIN_LABELS, right=False))
    return (c.groupby(["block_number", "bin"], observed=True)["included"]
            .agg(n="size", k="sum").reset_index())


def curve_from_counts(counts: pd.DataFrame, min_support: int) -> pd.DataFrame:
    g = counts.groupby("bin", observed=True)[["n", "k"]].sum().reset_index()
    g = g[g["n"] >= min_support]
    g["p"] = g["k"] / g["n"]
    return g


def bootstrap_delta(counts: pd.DataFrame, reps: int, rng: np.random.Generator,
                    min_support: int = 200) -> tuple[float, float, float]:
    blocks = counts["block_number"].unique()
    by_block = {b: g for b, g in counts.groupby("block_number", sort=False)}
    base = curve_from_counts(counts, min_support)
    point = float(base["p"].max() - base["p"].min()) if len(base) >= 2 else float("nan")
    draws = []
    for r in range(reps):
        idx = rng.integers(0, len(blocks), size=len(blocks))
        samp = pd.concat([by_block[blocks[i]] for i in idx], ignore_index=True)
        cur = curve_from_counts(samp, min_support)
        if len(cur) >= 2:
            draws.append(cur["p"].max() - cur["p"].min())
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--pos-blocks", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--boot-reps-delta", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261011)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = f"results/{a.name}"
    res: dict = {"window": a.name}

    # ---- item 1: position advantage, with CI
    blocks, meta, _ = load_window(a.days, with_mempool=False)
    all_blocks = blocks["block_number"].unique()
    sample = rng.choice(all_blocks, size=min(a.pos_blocks, len(all_blocks)),
                        replace=False)
    bsub = blocks[blocks["block_number"].isin(sample)].copy()
    ctrl = pd.concat([apply_random_control(g, draw_R(rng))
                      for _, g in bsub.groupby("block_number", sort=False)])
    bsub = bsub.assign(control_position=ctrl["control_position"])

    res["position"] = {}
    for arm, col in (("status_quo", "position"), ("random_control", "control_position")):
        st = per_block_pair_stats(bsub, col)
        pt, lo, hi = bootstrap_ratio(st["n_pairs"].to_numpy(), st["n_conc"].to_numpy(),
                                     a.reps, rng)
        res["position"][arm] = {"advantage": pt, "ci95": [lo, hi],
                               "n_blocks": int(len(st)),
                               "n_pairs": int(st["n_pairs"].sum())}
        st.to_csv(f"{out}/pairstats_{arm}.csv", index=False)

    # ---- items 2 and 3: inclusion channel
    cands = pd.read_csv(f"{out}/candidates.csv")
    fullness = pd.read_csv(f"{out}/fullness.csv").set_index("block_number")["fullness"]
    ref_median = pd.read_csv(f"{out}/reference_fee.csv").set_index("block_number")["priority_fee"]
    ref_basefee = cands.groupby("block_number")["base_fee"].first()

    # per-transaction first-candidacy view, to strip the length bias
    first_only = (cands.sort_values("block_number")
                  .drop_duplicates(subset="tx_hash", keep="first"))

    res["inclusion"] = {}
    for norm_name, norm in (("median_tip", ref_median), ("base_fee", ref_basefee)):
        for est_name, frame in (("per_row", cands), ("first_candidacy", first_only)):
            counts = binned_counts(frame, norm)
            if counts.empty:
                continue
            pt, lo, hi = bootstrap_delta(counts, a.boot_reps_delta, rng)
            cur = curve_from_counts(counts, 200)
            res["inclusion"][f"{norm_name}|{est_name}"] = {
                "delta_p": pt, "ci95": [lo, hi],
                "curve": [{"bin": str(r["bin"]), "p": float(r["p"]), "n": int(r["n"])}
                          for _, r in cur.iterrows()],
            }
            cur.to_csv(f"{out}/curve_{norm_name}_{est_name}.csv", index=False)

            # congestion strata, same normalizer and estimator
            f = frame.join(fullness.rename("fullness"), on="block_number")
            strata = []
            for nm, blo, bhi in (("<0.5x", 0.0, 0.5), ("0.5-0.9x", 0.5, 0.9),
                                 ("0.9-1.0x", 0.9, 1.0), ("1.0-1.5x", 1.0, 1.5),
                                 (">=1.5x", 1.5, 99.0)):
                sub = f[(f["fullness"] >= blo) & (f["fullness"] < bhi)]
                if len(sub) < 1000:
                    continue
                cc = binned_counts(sub, norm)
                if cc.empty:
                    continue
                p2, l2, h2 = bootstrap_delta(cc, 60, rng)
                strata.append({"stratum": nm, "delta_p": p2, "ci95": [l2, h2],
                               "n": int(len(sub))})
            res["inclusion"][f"{norm_name}|{est_name}"]["by_congestion"] = strata

    # ---- item 3 diagnostics: what is in the high-fee bins
    mem = pd.concat([pd.read_csv(f"{DERIVED}/mempool_{d}.csv",
                                usecols=["tx_hash", "included_at"],
                                dtype={"tx_hash": "string", "included_at": "float64"})
                     for d in a.days], ignore_index=True)
    ever = mem.dropna(subset=["included_at"]).set_index("tx_hash").index
    ever_set = set(ever)

    c = cands.join(ref_median.rename("norm"), on="block_number")
    c = c[(c["norm"] > 0) & (c["priority_fee"] > 0)]
    c = c.assign(bin=pd.cut(np.log2(c["priority_fee"] / c["norm"]),
                            bins=BIN_EDGES, labels=BIN_LABELS, right=False))
    diag = []
    for b, g in c.groupby("bin", observed=True):
        distinct = g["tx_hash"].nunique()
        never = sum(1 for h in g["tx_hash"].unique() if h not in ever_set)
        diag.append({"bin": str(b), "rows": int(len(g)), "distinct_tx": int(distinct),
                     "rows_per_distinct_tx": float(len(g) / distinct),
                     "never_included_anywhere_share": float(never / distinct)})
    res["high_fee_diagnostics"] = diag

    with open(f"{out}/results_final.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
