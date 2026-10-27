"""
inclusion_null.py

Adds the control the swap-adversary row was missing (WPPT 2026 review 8A).

The existing "null control" for the swap adversary permutes the ORDER inside each
block and leaves the INCLUDED SET untouched. The swap adversary decides mostly
from the inclusion bit, so that arm reproduces the measured value by
construction: it is an ablation of the position channel, not a control for the
inclusion channel.

This script runs the swap adversary on the clean candidate set (each
transaction's first candidacy, under one slot old) in three arms:

  status_quo       real inclusion, real block positions
  order_scrambled  real inclusion, keyed content-independent permutation of each
                   block's included set (the arm previously labelled "null control")
  inclusion_null   fee-independent inclusion AND order: in each block the
                   included labels are shuffled uniformly over that block's
                   candidates, so the included count is preserved but the set is
                   chosen without looking at fees, and the selected transactions
                   get a uniformly random order. Both channels are destroyed, so
                   the advantage should be near zero.

Intervals are a block bootstrap over the per-block trial outcomes (blocks are the
sampling unit, as in finalize.py).

Usage (add --positions when data/derived/blocks_<day>.csv exist):
    python3 code/inclusion_null.py --name congested --days 20240319 20240320 20240321
    python3 code/inclusion_null.py --name quiet     --days 20251104 20251105 20251106
Writes results/<name>/results_swap_controls.json
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
from run_window import load_window, SLOT_MS  # noqa: E402
from finalize import bootstrap_ratio  # noqa: E402


def clean_candidates(cands: pd.DataFrame) -> pd.DataFrame:
    first = (cands.sort_values("block_number")
             .drop_duplicates(subset="tx_hash", keep="first"))
    return first[first["age_ms"] < SLOT_MS].reset_index(drop=True)


def swap_adversary_blocks(cands: pd.DataFrame, pos_map: dict, reference: dict,
                          selection: str, rng: np.random.Generator,
                          n_trials: int) -> pd.DataFrame:
    """Same decision rule as run_window.swap_adversary, but returns per-block
    (n_ok, n_correct) so the result can be block-bootstrapped."""
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

    bids = np.array(list(per_block.keys()))
    picks = rng.choice(len(bids), size=n_trials, replace=True)
    coins = rng.integers(0, 2, size=n_trials)
    ok = np.zeros(len(bids), dtype=np.int64)
    cor = np.zeros(len(bids), dtype=np.int64)

    for k in range(n_trials):
        bi = picks[k]
        fee, inc, snd, pos, hi, lo = per_block[bids[bi]]
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
        ok[bi] += 1
        cor[bi] += int(guess_i == hi_is_i)
    df = pd.DataFrame({"block_number": bids, "n": ok, "k": cor})
    return df[df["n"] > 0]


def inclusion_null_arm(cands: pd.DataFrame, rng: np.random.Generator
                       ) -> tuple[pd.DataFrame, dict]:
    """Shuffle the included labels within each block (count preserved, choice
    fee-independent) and give the selected transactions a random order."""
    parts, pos_map = [], {}
    for _, g in cands.groupby("block_number", sort=False):
        lab = g["included"].to_numpy().astype(bool).copy()
        rng.shuffle(lab)
        g = g.assign(included=lab)
        hs = g.loc[lab, "tx_hash"].to_numpy()
        for h, p in zip(hs, rng.permutation(len(hs))):
            pos_map[h] = int(p)
        parts.append(g)
    return pd.concat(parts, ignore_index=True), pos_map


def random_order(cands: pd.DataFrame, rng: np.random.Generator) -> dict:
    pos_map = {}
    for _, g in cands.groupby("block_number", sort=False):
        hs = g.loc[g["included"].astype(bool), "tx_hash"].to_numpy()
        for h, p in zip(hs, rng.permutation(len(hs))):
            pos_map[h] = int(p)
    return pos_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--positions", action="store_true",
                    help="also run the status-quo arm (needs the raw block dumps)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = f"results/{a.name}"
    cands = clean_candidates(pd.read_csv(f"{out}/candidates.csv"))
    reference = (pd.read_csv(f"{out}/reference_fee.csv")
                 .set_index("block_number")["priority_fee"].to_dict())
    arms = {}
    if a.positions:
        # needs data/derived/blocks_<day>.csv (per-transaction block positions)
        blocks, _, _ = load_window(a.days, with_mempool=False,
                                   only_blocks=set(cands["block_number"].unique()))
        ctrl = pd.concat([apply_random_control(g, draw_R(rng))
                          for _, g in blocks.groupby("block_number", sort=False)])
        blocks = blocks.assign(control_position=ctrl["control_position"])

        def pmap(col: str) -> dict:
            s = blocks.set_index("tx_hash")[col]
            return s[~s.index.duplicated()].to_dict()
        arms["status_quo"] = (cands, pmap("position"))
        arms["order_scrambled"] = (cands, pmap("control_position"))
    else:
        # Without the raw per-transaction dumps, the order-scrambled arm is run
        # with a uniform random order over each block's included candidates.
        # For the swap adversary only the relative order of two included
        # candidates matters, and that is uniform under either construction.
        arms["order_scrambled"] = (cands, random_order(cands, rng))

    # What the September run actually computed: no position ever matched (the
    # Blockchair hashes lacked the 0x prefix the mempool hashes carry), so every
    # pair in which both transactions were included was skipped. That is the
    # inclusion bit's advantage CONDITIONAL on the pair not being both included.
    # Reported for the record; it is not a lower bound on the game advantage.
    arms["inclusion_conditional_not_both_included"] = (cands, {})

    null_cands, null_pos = inclusion_null_arm(cands, rng)
    arms["inclusion_null"] = (null_cands, null_pos)

    res = {"window": a.name, "positions_available": bool(a.positions), "n_clean_candidates": int(len(cands)),
           "n_blocks": int(cands["block_number"].nunique()),
           "trials": a.trials, "bootstrap_reps": a.reps, "seed": a.seed, "arms": {}}
    # outcome mix of extremal pairs, for interpreting the conditional arm
    mix = {"differ": 0, "both_included": 0, "both_excluded": 0}
    g = cands.copy()
    for b, gg in g.groupby("block_number", sort=False):
        ref = reference.get(b, 0.0)
        if not ref or ref <= 0:
            continue
        f = gg["priority_fee"].to_numpy(); inc = gg["included"].to_numpy().astype(bool)
        hi, lo = inc[f >= 2 * ref], inc[f <= ref / 2]
        if len(hi) == 0 or len(lo) == 0:
            continue
        h1, l1 = hi.mean(), lo.mean()
        w = len(hi) * len(lo)
        mix["both_included"] += w * h1 * l1
        mix["both_excluded"] += w * (1 - h1) * (1 - l1)
        mix["differ"] += w * (h1 * (1 - l1) + (1 - h1) * l1)
    tot = sum(mix.values())
    res["extremal_pair_outcome_mix"] = {k: float(v / tot) for k, v in mix.items()}
    print("extremal outcome mix", res["extremal_pair_outcome_mix"], flush=True)

    for arm, (cc, pm) in arms.items():
        for sel in ("extremal", "random"):
            per = swap_adversary_blocks(cc, pm, reference, sel, rng, a.trials)
            pt, lo, hi = bootstrap_ratio(per["n"].to_numpy(), per["k"].to_numpy(),
                                         a.reps, rng)
            acc = per["k"].sum() / per["n"].sum()
            res["arms"][f"{arm}_{sel}"] = {
                "advantage": pt, "ci95": [lo, hi],
                "accuracy": float(acc), "n_trials_scored": int(per["n"].sum()),
                "n_blocks": int(len(per))}
            print(arm, sel, round(pt, 4), [round(lo, 4), round(hi, 4)],
                  int(per["n"].sum()), flush=True)

    with open(f"{out}/results_swap_controls.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
