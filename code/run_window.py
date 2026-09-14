"""
run_window.py

Runs the ordering-leakage measurement over one window (a list of extracted days).

    python3 code/run_window.py --name congested --days 20240319 20240320 20240321
    python3 code/run_window.py --name quiet     --days 20251104 20251105 20251106

Reads data/derived/{blocks,blocks_meta,mempool}_<day>.csv produced by
code/extract_day.sh, writes results/<name>/.

Candidate construction, replacing the naive loop in ordering_leakage.py:
for a block b at time t_b, the candidate set is every mempool transaction first
seen in [t_b - W, t_b] that had not already been included in an earlier block.
The window slice is taken with searchsorted over a sorted first_seen array, so
cost per block is proportional to the arrivals in W, not to the whole feed.

Effective priority fee is min(gas_tip_cap, fee_cap - base_fee_of_that_block),
which is what a builder actually earns. Transactions whose fee cap sits below
the block's base fee cannot be included at any tip; they are counted and
reported separately rather than silently dropped or silently kept, because
including them inflates the bottom bin for a reason that is real but different
from the one the theorem is about.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordering_leakage import (  # noqa: E402
    apply_random_control, draw_R, position_advantage, PairStats,
)

DERIVED = "data/derived"
SLOT_MS = 12_000


def load_window(days: list[str], with_mempool: bool = True,
                only_blocks: set | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    blocks = pd.concat(
        [pd.read_csv(f"{DERIVED}/blocks_{d}.csv",
                     usecols=["block_number", "tx_hash", "position", "sender",
                              "priority_fee", "base_fee"],
                     dtype={"block_number": "int64", "tx_hash": "string",
                            "position": "int32", "sender": "string",
                            "priority_fee": "float64", "base_fee": "float64"})
         for d in days], ignore_index=True)
    if only_blocks is not None:
        blocks = blocks[blocks["block_number"].isin(only_blocks)].reset_index(drop=True)
    meta = pd.concat(
        [pd.read_csv(f"{DERIVED}/blocks_meta_{d}.csv") for d in days],
        ignore_index=True).sort_values("block_number").reset_index(drop=True)
    if not with_mempool:
        return blocks, meta, None
    mem = pd.concat(
        [pd.read_csv(f"{DERIVED}/mempool_{d}.csv",
                     usecols=["tx_hash", "first_seen_ms", "sender",
                              "priority_fee", "fee_cap", "included_at", "nonce"],
                     dtype={"tx_hash": "string", "first_seen_ms": "int64",
                            "sender": "string", "priority_fee": "float64",
                            "fee_cap": "float64", "included_at": "float64",
                            "nonce": "float64"})
         for d in days], ignore_index=True)
    mem = mem.dropna(subset=["first_seen_ms", "priority_fee", "fee_cap"])
    mem = mem.sort_values("first_seen_ms").reset_index(drop=True)
    return blocks, meta, mem


def load_mempool_only(days: list[str]) -> pd.DataFrame:
    mem = pd.concat(
        [pd.read_csv(f"{DERIVED}/mempool_{d}.csv",
                     usecols=["tx_hash", "first_seen_ms", "sender",
                              "priority_fee", "fee_cap", "included_at", "nonce"],
                     dtype={"tx_hash": "string", "first_seen_ms": "int64",
                            "sender": "string", "priority_fee": "float64",
                            "fee_cap": "float64", "included_at": "float64",
                            "nonce": "float64"})
         for d in days], ignore_index=True)
    mem = mem.dropna(subset=["first_seen_ms", "priority_fee", "fee_cap"])
    return mem.sort_values("first_seen_ms").reset_index(drop=True)


def build_candidates_fast(meta: pd.DataFrame, mem: pd.DataFrame,
                          block_ids: np.ndarray, window_blocks: int = 25
                          ) -> tuple[pd.DataFrame, dict]:
    W = window_blocks * SLOT_MS
    seen = mem["first_seen_ms"].to_numpy()
    inc = mem["included_at"].to_numpy()          # NaN when never included
    tip = mem["priority_fee"].to_numpy()
    cap = mem["fee_cap"].to_numpy()
    snd = mem["sender"].to_numpy()

    mb = meta["block_number"].to_numpy()
    mt = meta["timestamp_ms"].to_numpy()
    mf = meta["base_fee"].to_numpy()
    order = np.argsort(mb)
    mb, mt, mf = mb[order], mt[order], mf[order]

    hashes = mem["tx_hash"].to_numpy()
    nonce = mem["nonce"].to_numpy()
    out_b, out_h, out_s, out_f, out_i, out_bf, out_n = [], [], [], [], [], [], []
    out_age = []
    stats = {"blocks": 0, "below_base_fee": 0, "candidates": 0}

    for b in block_ids:
        k = np.searchsorted(mb, b)
        if k >= len(mb) or mb[k] != b:
            continue
        t, bf = int(mt[k]), float(mf[k])
        lo, hi = np.searchsorted(seen, [t - W, t])
        if hi <= lo:
            continue
        inc_sl = inc[lo:hi]
        pending = np.isnan(inc_sl) | (inc_sl >= b)
        if not pending.any():
            continue
        idx = np.arange(lo, hi)[pending]

        eff = np.minimum(tip[idx], cap[idx] - bf)
        eligible = eff > 0
        stats["below_base_fee"] += int((~eligible).sum())
        idx, eff = idx[eligible], eff[eligible]
        if len(idx) == 0:
            continue

        out_b.append(np.full(len(idx), b, dtype=np.int64))
        out_h.append(hashes[idx])
        out_s.append(snd[idx])
        out_f.append(eff)
        out_i.append(inc[idx] == b)
        out_bf.append(np.full(len(idx), bf))
        out_n.append(nonce[idx])
        out_age.append(t - seen[idx])
        stats["blocks"] += 1
        stats["candidates"] += len(idx)

    rows = out_b
    if not rows:
        return pd.DataFrame(), stats
    cands = pd.DataFrame({
        "block_number": np.concatenate(out_b),
        "tx_hash": np.concatenate(out_h),
        "sender": np.concatenate(out_s),
        "priority_fee": np.concatenate(out_f),
        "included": np.concatenate(out_i),
        "base_fee": np.concatenate(out_bf),
        "nonce": np.concatenate(out_n),
        "age_ms": np.concatenate(out_age),
    })
    # One candidate per (block, sender, nonce). A fee-bumped replacement appears in
    # the feed as several hashes sharing a sender and nonce, of which at most one
    # can ever be included; counting them all puts a crowd of permanently excluded
    # high-fee entries into the top bins and makes inclusion look like it FALLS
    # with fee. Keep the included version when there is one, otherwise the
    # highest-fee version, which is what a builder would have considered.
    before = len(cands)
    cands = (cands.sort_values(["included", "priority_fee"], ascending=[False, False])
             .groupby(["block_number", "sender", "nonce"], as_index=False, sort=False)
             .first())
    stats["rows_before_replacement_dedupe"] = int(before)
    stats["rows_after_replacement_dedupe"] = int(len(cands))
    return cands, stats


def reference_fee_map(blocks: pd.DataFrame) -> pd.Series:
    """The block's going rate: the MEDIAN effective priority fee among the
    transactions it included.

    The obvious choice, the minimum included tip, is useless on real data: about
    three quarters of blocks in both windows include at least one zero-tip
    transaction, so the minimum is 0 in most blocks. That fact is itself worth
    reporting, since it is direct evidence that inclusion on today's Ethereum is
    not a pure fee auction: private order flow and builder side-agreements put
    zero-tip transactions on chain. The median is robust to that mass at zero.
    """
    return blocks.groupby("block_number")["priority_fee"].median()


def inclusion_curve(cands: pd.DataFrame, reference: pd.Series,
                    min_support: int = 200) -> tuple[pd.DataFrame, float]:
    c = cands.join(reference.rename("ref_fee"), on="block_number")
    c = c[c["ref_fee"] > 0]
    if c.empty:
        return pd.DataFrame(), float("nan")
    c = c.assign(log_ratio=np.log2(c["priority_fee"] / c["ref_fee"]))
    edges = [-np.inf, -4, -2, -1, -0.25, 0.25, 1, 2, 4, np.inf]
    labels = ["<-4", "[-4,-2)", "[-2,-1)", "[-1,-.25)", "[-.25,.25)",
              "[.25,1)", "[1,2)", "[2,4)", ">=4"]
    c = c.assign(bin=pd.cut(c["log_ratio"], bins=edges, labels=labels, right=False))
    curve = (c.groupby("bin", observed=True)["included"].agg(["mean", "count"])
             .reset_index().rename(columns={"mean": "p", "count": "n"}))
    curve = curve[curve["n"] >= min_support]
    if len(curve) < 2:
        return curve, float("nan")
    return curve, float(curve["p"].max() - curve["p"].min())


def stratified_delta_simple(cands: pd.DataFrame, reference: pd.Series,
                            fullness: pd.Series, min_support: int = 200) -> pd.DataFrame:
    c = cands.join(fullness.rename("fullness"), on="block_number")
    out = []
    for name, lo, hi in (("gas<0.5x target", 0.0, 0.5),
                         ("0.5-0.9x target", 0.5, 0.9),
                         ("0.9-1.0x target", 0.9, 1.0),
                         ("1.0-1.5x target", 1.0, 1.5),
                         (">=1.5x target", 1.5, 99.0)):
        sub = c[(c["fullness"] >= lo) & (c["fullness"] < hi)]
        if len(sub) < min_support * 2:
            continue
        curve, d = inclusion_curve(sub, reference, min_support=min_support)
        out.append({"stratum": name, "delta_p": d, "n_candidates": int(len(sub)),
                    "n_blocks": int(sub["block_number"].nunique())})
    return pd.DataFrame(out)


def swap_adversary(cands: pd.DataFrame, pos_map: dict, reference: dict,
                   position_col_missing_ok: bool, selection: str,
                   rng: np.random.Generator, n_trials: int = 200_000) -> PairStats:
    """Explicit swap adversary. See the design file for the decision rule.

    Everything is precomputed into plain numpy arrays per block, because indexing
    a pandas frame inside the trial loop is two orders of magnitude slower and
    turns a two-second job into a ten-minute one.
    """
    per_block = {}
    for b, g in cands.groupby("block_number", sort=False):
        if len(g) < 2:
            continue
        fee = g["priority_fee"].to_numpy()
        inc = g["included"].to_numpy().astype(bool)
        snd = g["sender"].to_numpy()
        pos = np.array([pos_map.get(h, -1) for h in g["tx_hash"].to_numpy()])
        ref = reference.get(b, 0.0)
        if selection == "extremal":
            if not ref or ref <= 0:
                continue
            hi = np.flatnonzero(fee >= 2 * ref)
            lo = np.flatnonzero(fee <= ref / 2)
            if len(hi) == 0 or len(lo) == 0:
                continue
            per_block[b] = (fee, inc, snd, pos, hi, lo)
        else:
            per_block[b] = (fee, inc, snd, pos, None, None)
    if not per_block:
        return PairStats(0, 0)

    bids = np.array(list(per_block.keys()))
    picks = rng.choice(len(bids), size=n_trials, replace=True)
    coins = rng.integers(0, 2, size=n_trials)
    n_ok = n_correct = 0

    for k in range(n_trials):
        fee, inc, snd, pos, hi, lo = per_block[bids[picks[k]]]
        if hi is not None:
            i = hi[rng.integers(len(hi))]
            j = lo[rng.integers(len(lo))]
        else:
            i = rng.integers(len(fee))
            j = rng.integers(len(fee))
            if i == j:
                continue
        if snd[i] == snd[j] or fee[i] == fee[j]:
            continue

        hi_is_i = fee[i] > fee[j]
        if inc[i] != inc[j]:
            guess_i = bool(inc[i])
        elif inc[i]:
            if pos[i] < 0 or pos[j] < 0:
                continue
            guess_i = pos[i] < pos[j]
        else:
            guess_i = bool(coins[k])
        n_ok += 1
        n_correct += int(guess_i == hi_is_i)
    return PairStats(n_ok, n_correct)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--step", required=True,
                    choices=["position", "candidates", "metrics"])
    ap.add_argument("--pos-blocks", type=int, default=3000)
    ap.add_argument("--cand-blocks", type=int, default=1500)
    ap.add_argument("--horizon-blocks", type=int, default=25,
                    help="how long a pending transaction stays in the candidate set")
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20261011)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out_dir = f"results/{a.name}"
    os.makedirs(out_dir, exist_ok=True)

    def save(part: str, obj) -> None:
        with open(f"{out_dir}/{part}.json", "w") as fh:
            json.dump(obj, fh, indent=2, default=float)
        print(json.dumps(obj, indent=2, default=float)[:4000])

    if a.step == "position":
        blocks, meta, _ = load_window(a.days, with_mempool=False)
        res = {"window": a.name, "days": a.days,
               "n_blocks_total": int(blocks["block_number"].nunique()),
               "n_tx_rows": int(len(blocks)),
               "median_base_fee_gwei": float(meta["base_fee"].median() / 1e9),
               "median_gas_vs_target": float((meta["gas_used"] / (meta["gas_limit"] / 2.0)).median()),
               "blocks_with_zero_tip_share": float(
                   (blocks.groupby("block_number")["priority_fee"].min() == 0).mean()),
               "position": {}}
        all_blocks = blocks["block_number"].unique()
        sample = rng.choice(all_blocks, size=min(a.pos_blocks, len(all_blocks)),
                            replace=False)
        bsub = blocks[blocks["block_number"].isin(sample)].copy()
        ctrl = pd.concat([apply_random_control(g, draw_R(rng))
                          for _, g in bsub.groupby("block_number", sort=False)])
        bsub = bsub.assign(control_position=ctrl["control_position"])
        for arm, col in (("status_quo", "position"),
                         ("random_control", "control_position")):
            ps = position_advantage(bsub, position_col=col, seed=a.seed)
            res["position"][arm] = {"accuracy": ps.accuracy,
                                    "advantage": ps.advantage,
                                    "n_pairs": ps.n_pairs,
                                    "n_blocks": int(len(sample))}
        reference_fee_map(blocks).to_csv(f"{out_dir}/reference_fee.csv")
        save("results_position", res)
        return

    if a.step == "candidates":
        meta = pd.concat([pd.read_csv(f"{DERIVED}/blocks_meta_{d}.csv") for d in a.days],
                         ignore_index=True).sort_values("block_number").reset_index(drop=True)
        mem = load_mempool_only(a.days)
        all_blocks = meta["block_number"].to_numpy()
        sample = np.sort(rng.choice(all_blocks,
                                    size=min(a.cand_blocks, len(all_blocks)),
                                    replace=False))
        cands, stats = build_candidates_fast(meta, mem, sample,
                                            window_blocks=a.horizon_blocks)
        stats["horizon_blocks"] = a.horizon_blocks
        cands.to_csv(f"{out_dir}/candidates.csv", index=False)
        # congestion is measured against the EIP-1559 TARGET (half the limit),
        # not the limit: a block at the target is exactly break-even for the base
        # fee, and blocks above it push the base fee up. Measuring against the
        # limit makes a fully congested chain look half empty.
        (meta.set_index("block_number")["gas_used"]
         / (meta.set_index("block_number")["gas_limit"] / 2.0)).rename("fullness") \
            .to_csv(f"{out_dir}/fullness.csv")
        save("results_candidates", {"window": a.name, "candidate_stats": stats})
        return

    # step == metrics
    cands = pd.read_csv(f"{out_dir}/candidates.csv")
    reference = pd.read_csv(f"{out_dir}/reference_fee.csv").set_index("block_number")["priority_fee"]
    fullness = pd.read_csv(f"{out_dir}/fullness.csv").set_index("block_number")["fullness"]
    blocks, _, _ = load_window(a.days, with_mempool=False,
                               only_blocks=set(cands["block_number"].unique()))

    curve, delta = inclusion_curve(cands, reference)
    curve.to_csv(f"{out_dir}/inclusion_curve.csv", index=False)
    strat = stratified_delta_simple(cands, reference, fullness)
    strat.to_csv(f"{out_dir}/delta_p_by_fullness.csv", index=False)

    ctrl = pd.concat([apply_random_control(g, draw_R(rng))
                      for _, g in blocks.groupby("block_number", sort=False)])
    blocks = blocks.assign(control_position=ctrl["control_position"])
    ref_map = reference.to_dict()
    game = {}
    for arm, col in (("status_quo", "position"),
                     ("random_control", "control_position")):
        pos_map = blocks.set_index("tx_hash")[col]
        pos_map = pos_map[~pos_map.index.duplicated()].to_dict()
        for sel in ("extremal", "random"):
            gs = swap_adversary(cands, pos_map, ref_map, True, sel, rng, a.trials)
            game[f"{arm}_{sel}"] = {"accuracy": gs.accuracy,
                                    "advantage": gs.advantage,
                                    "n_trials": gs.n_pairs}
    save("results_metrics", {"window": a.name, "delta_p": delta,
                             "inclusion_curve": curve.to_dict(orient="records"),
                             "delta_p_by_fullness": strat.to_dict(orient="records"),
                             "game": game})


if __name__ == "__main__":
    main()
