"""
ordering_leakage.py

Measurement code for contribution (c) of the WPPT 2026 ordering-privacy abstract.

WHAT THIS MEASURES
------------------
An explicit adversary's advantage in the __NAME__ game, estimated on historical
Ethereum data replayed through two ordering rules:

  arm "status_quo"     : transactions keep the position they actually had on chain
  arm "random_control" : positions are replaced by a content-independent random
                         permutation. This is a NULL CONTROL, not a proposal: it is
                         the order-blind corner of the theory (Proposition 8), and it
                         exists so the position estimator can be shown to read zero
                         when the order carries no fee information.

Three quantities are produced:

  A. position advantage   Adv_pos  : can the position alone reveal which of two
                                     senders paid the higher priority fee
  B. inclusion advantage  Delta_p  : the responsiveness gap of the inclusion rule,
                                     |p(f_hi) - p(f_lo)|, stratified by congestion
  C. game advantage       Adv_game : accuracy of an explicit swap adversary that
                                     sees inclusion and position, minus a coin flip

The status_quo arm is an OBSERVATIONAL measurement of deployed Ethereum: no proposed
ordering rule is simulated, and nothing here is a counterfactual replay. The
random_control arm is a synthetic null control on the same data, and must be labelled
as such. Do not describe either as a study of any unshipped proposal.

Because the adversary is explicit and suboptimal, every measured advantage is a
LOWER bound on epsilon. That is the direction the theory needs (epsilon >= Delta_p),
so this is the right way round, but it must be stated as a lower bound, never as
"the leakage is X".

INPUT SCHEMAS
-------------
blocks.csv (one row per included transaction)
    block_number      int
    tx_hash           str, 0x-prefixed 32-byte hex
    position          int, 0-based index within the block
    sender            str, 0x-prefixed address
    priority_fee      int, effective priority fee per gas, in wei
    base_fee          int, block base fee per gas, in wei (repeated per row)
    gas_used          int, gas used by this transaction
    block_gas_limit   int, repeated per row

mempool.csv (one row per transaction observed in the public mempool)
    tx_hash           str
    first_seen_ms     int, unix milliseconds
    sender            str
    priority_fee      int, max priority fee per gas offered, in wei

blocks_meta.csv (one row per block)
    block_number      int
    timestamp_ms      int
    base_fee          int
    gas_used          int
    gas_limit         int

Metric A needs only blocks.csv. Metrics B and C need all three.

USAGE
-----
    python3 ordering_leakage.py --selftest
    python3 ordering_leakage.py --blocks blocks.csv --meta blocks_meta.csv \
                                --mempool mempool.csv --out results/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

RNG_DEFAULT_SEED = 20261011


# ---------------------------------------------------------------------------
# Null control: a content-independent permutation of the included set
# ---------------------------------------------------------------------------

def _control_sort_key(tx_hash_hex: str, R: int) -> tuple[int, int]:
    """Sort key for the null control: (H(tx) XOR R) ascending, H(tx) ascending.

    The construction is a keyed pseudorandom permutation of the included set that is
    independent of every transaction's content, which is all the control needs to be.
    It is not a model of any particular proposal.
    """
    h = bytes.fromhex(tx_hash_hex[2:] if tx_hash_hex.startswith("0x") else tx_hash_hex)
    if len(h) != 32:
        raise ValueError(f"expected a 32-byte tx hash, got {len(h)} bytes")
    full = int.from_bytes(h, "big")
    primary = (full >> 128) ^ R
    return (primary, full)

def apply_random_control(block_df: pd.DataFrame, R: int) -> pd.DataFrame:
    """Return block_df with a new column control_position."""
    keys = [_control_sort_key(h, R) for h in block_df["tx_hash"]]
    order = np.argsort(np.array([k[0] for k in keys], dtype=object), kind="stable")
    out = block_df.iloc[order].copy()
    out["control_position"] = np.arange(len(out))
    return out.sort_index()


def draw_R(rng: np.random.Generator) -> int:
    """A 16-byte key for the control permutation, drawn uniformly per block."""
    return int.from_bytes(rng.bytes(16), "big")


# ---------------------------------------------------------------------------
# Metric A: position advantage
# ---------------------------------------------------------------------------

@dataclass
class PairStats:
    n_pairs: int
    n_concordant: int

    @property
    def accuracy(self) -> float:
        return self.n_concordant / self.n_pairs if self.n_pairs else float("nan")

    @property
    def advantage(self) -> float:
        return abs(2 * self.accuracy - 1) if self.n_pairs else float("nan")


def _block_pair_stats(pos: np.ndarray, fee: np.ndarray, sender: np.ndarray,
                      max_txs_exact: int, rng: np.random.Generator,
                      n_sampled_pairs: int) -> tuple[int, int]:
    """Count (pairs, concordant) within one block.

    A pair is admissible when the two transactions have different senders and
    different fees, which is exactly the Adm_assign swap the game uses.
    Concordant means the higher fee sits at the earlier position.
    """
    k = len(pos)
    if k < 2:
        return 0, 0

    if k <= max_txs_exact:
        i, j = np.triu_indices(k, k=1)
    else:
        i = rng.integers(0, k, size=n_sampled_pairs)
        j = rng.integers(0, k, size=n_sampled_pairs)
        keep = i != j
        i, j = i[keep], j[keep]

    ok = (sender[i] != sender[j]) & (fee[i] != fee[j])
    i, j = i[ok], j[ok]
    if len(i) == 0:
        return 0, 0

    higher_fee_first = (fee[i] > fee[j]) == (pos[i] < pos[j])
    return int(len(i)), int(np.count_nonzero(higher_fee_first))


def position_advantage(blocks: pd.DataFrame, position_col: str = "position",
                       max_txs_exact: int = 400, n_sampled_pairs: int = 50_000,
                       seed: int = RNG_DEFAULT_SEED) -> PairStats:
    rng = np.random.default_rng(seed)
    total = concordant = 0
    for _, blk in blocks.groupby("block_number", sort=False):
        n, c = _block_pair_stats(
            blk[position_col].to_numpy(),
            blk["priority_fee"].to_numpy(),
            blk["sender"].to_numpy(),
            max_txs_exact, rng, n_sampled_pairs,
        )
        total += n
        concordant += c
    return PairStats(total, concordant)


# ---------------------------------------------------------------------------
# Candidate sets and Metric B: inclusion advantage
# ---------------------------------------------------------------------------

def build_candidates(blocks: pd.DataFrame, meta: pd.DataFrame, mempool: pd.DataFrame,
                     max_wait_blocks: int = 25, slot_ms: int = 12_000) -> pd.DataFrame:
    """For each block b, the candidate set is every mempool transaction first seen
    before b's timestamp that had not been included in a block before b, limited to
    transactions that are eventually included within max_wait_blocks or never seen
    again. Returns one row per (block, candidate) with an `included` flag.

    This is the step where a reader will push hardest, and rightly: a transaction
    absent from our mempool feed is not necessarily absent from the builder's view,
    and private order flow never appears in a public feed at all. Both directions of
    that error are discussed in the measurement design file. Do not present the
    candidate set as the builder's true choice set.
    """
    meta = meta.sort_values("block_number").reset_index(drop=True)
    inclusion_block = (blocks.groupby("tx_hash")["block_number"].min()
                       .rename("included_in").reset_index())
    mp = mempool.merge(inclusion_block, on="tx_hash", how="left")

    rows = []
    for _, m in meta.iterrows():
        b = int(m["block_number"])
        t = int(m["timestamp_ms"])
        stale_before = t - max_wait_blocks * slot_ms
        seen = mp[(mp["first_seen_ms"] <= t) & (mp["first_seen_ms"] >= stale_before)]
        pending = seen[(seen["included_in"].isna()) | (seen["included_in"] >= b)]
        pending = pending[(pending["included_in"].isna())
                          | (pending["included_in"] <= b + max_wait_blocks)]
        if pending.empty:
            continue
        sub = pending[["tx_hash", "sender", "priority_fee"]].copy()
        sub["block_number"] = b
        sub["included"] = (pending["included_in"] == b).fillna(False).to_numpy()
        sub["base_fee"] = int(m["base_fee"])
        sub["fullness"] = float(m["gas_used"]) / float(m["gas_limit"])
        sub["mempool_depth"] = len(pending)
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["tx_hash", "sender", "priority_fee", "block_number",
                                     "included", "base_fee", "fullness", "mempool_depth"])
    return pd.concat(rows, ignore_index=True)


def _fee_bins(g: pd.DataFrame, n_bins: int) -> pd.Series:
    """Within-block fee percentile bin. The theorem's p(f) holds the other fees fixed,
    so the conditioning variable is the fee's rank inside its own candidate set, not
    its absolute value."""
    ranks = g["priority_fee"].rank(method="average", pct=True)
    return np.minimum((ranks * n_bins).astype(int), n_bins - 1)


def clearing_fee(blocks: pd.DataFrame) -> pd.Series:
    """The block's clearing fee: the lowest priority fee it actually included.

    This is the endogenous threshold the theory cares about. Binning candidates by
    fee / clearing_fee is what resolves the step in p that a coarse percentile bin
    cannot see, because under a deep backlog the clearing threshold sits inside the
    top percent or two of the candidate set.
    """
    return blocks.groupby("block_number")["priority_fee"].min().rename("clearing_fee")


def inclusion_advantage_by_clearing(candidates: pd.DataFrame, blocks: pd.DataFrame,
                                    min_support: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary estimator: p as a function of log2(fee / clearing_fee of that block).

    Delta_p is taken as p(top bin) - p(bottom bin) over bins meeting min_support,
    which is the empirical analogue of the supremum in Definition S.
    """
    cf = clearing_fee(blocks)
    c = candidates.merge(cf, on="block_number", how="inner")
    c = c[(c["priority_fee"] > 0) & (c["clearing_fee"] > 0)].copy()
    if c.empty:
        return pd.DataFrame(), pd.DataFrame()

    c["log_ratio"] = np.log2(c["priority_fee"] / c["clearing_fee"])
    edges = [-np.inf, -4, -2, -1, -0.25, 0.25, 1, 2, 4, np.inf]
    labels = ["<-4", "[-4,-2)", "[-2,-1)", "[-1,-.25)", "[-.25,.25)",
              "[.25,1)", "[1,2)", "[2,4)", ">=4"]
    c["ratio_bin"] = pd.cut(c["log_ratio"], bins=edges, labels=labels, right=False)

    curve = (c.groupby("ratio_bin", observed=True)["included"]
             .agg(["mean", "count"]).reset_index()
             .rename(columns={"mean": "p", "count": "n"}))
    curve = curve[curve["n"] >= min_support]
    if len(curve) < 2:
        return pd.DataFrame(), curve

    summary = pd.DataFrame([{
        "estimator": "clearing_ratio",
        "delta_p": float(curve["p"].max() - curve["p"].min()),
        "p_max_bin": str(curve.loc[curve["p"].idxmax(), "ratio_bin"]),
        "p_min_bin": str(curve.loc[curve["p"].idxmin(), "ratio_bin"]),
        "n_candidates": int(curve["n"].sum()),
    }])
    return summary, curve


def inclusion_advantage(candidates: pd.DataFrame, n_bins: int = 50,
                        min_support: int = 100,
                        congestion_bins: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95, 1.01)
                        ) -> pd.DataFrame:
    """Estimate p by within-block fee bin, stratified by block fullness.

    Delta_p for a stratum is max(p) - min(p) over bins meeting min_support.
    """
    c = candidates.copy()
    c["fee_bin"] = c.groupby("block_number", group_keys=False).apply(
        lambda g: _fee_bins(g, n_bins))
    c["stratum"] = pd.cut(c["fullness"], bins=list(congestion_bins), right=False,
                          labels=[f"fullness[{congestion_bins[i]:.2f},{congestion_bins[i+1]:.2f})"
                                  for i in range(len(congestion_bins) - 1)])

    g = (c.groupby(["stratum", "fee_bin"], observed=True)["included"]
         .agg(["mean", "count"]).reset_index()
         .rename(columns={"mean": "p", "count": "n"}))
    g = g[g["n"] >= min_support]

    out = []
    for stratum, sub in g.groupby("stratum", observed=True):
        if len(sub) < 2:
            continue
        out.append({
            "stratum": str(stratum),
            "delta_p": float(sub["p"].max() - sub["p"].min()),
            "p_top_bin": float(sub.sort_values("fee_bin")["p"].iloc[-1]),
            "p_bottom_bin": float(sub.sort_values("fee_bin")["p"].iloc[0]),
            "n_candidates": int(sub["n"].sum()),
            "n_bins_used": int(len(sub)),
        })
    return pd.DataFrame(out), g


# ---------------------------------------------------------------------------
# Metric C: the explicit swap adversary
# ---------------------------------------------------------------------------

def swap_adversary(candidates: pd.DataFrame, blocks: pd.DataFrame,
                   position_col: str, n_trials: int = 200_000,
                   seed: int = RNG_DEFAULT_SEED,
                   pair_selection: str = "extremal") -> PairStats:
    """Estimate the accuracy of this adversary:

        given two candidate wrappers in the same block, from different senders,
        carrying different fees, and knowing only what View_ord contains,
        guess which sender carried the higher fee.

        rule: if exactly one of the two was included, guess that one
              if both were included, guess the one at the earlier position
              if neither was included, flip a coin

    This is a concrete, suboptimal adversary, so its advantage is a lower bound
    on epsilon. Under the eip7956 arm the position tiebreak is independent of the
    fee by construction, so the "both included" case degenerates to a coin flip;
    that is the whole point of the comparison.

    pair_selection:
      "extremal" - the adversary picks the challenge pair to straddle the block's
                   clearing fee: one candidate paying at least 2x it, one paying at
                   most half of it. This is the empirical analogue of the supremum
                   in the theorem, and it is what the game allows, since the
                   adversary chooses T_0 and T_1. It assumes the adversary knows
                   roughly what the going clearing price is, which any searcher does.
      "random"   - a uniformly random admissible pair, which answers the different
                   and weaker question of how exposed a typical pair of users is.
                   Report this too; the gap between the two is informative.
    """
    rng = np.random.default_rng(seed)
    pos_lookup = blocks.set_index("tx_hash")[position_col].to_dict()

    cf = clearing_fee(blocks).to_dict()
    by_block = {b: g for b, g in candidates.groupby("block_number", sort=False) if len(g) >= 2}
    if not by_block:
        return PairStats(0, 0)
    block_ids = np.array(list(by_block.keys()))

    n_ok = n_correct = 0
    for b in rng.choice(block_ids, size=min(n_trials, 40 * len(block_ids)), replace=True):
        g = by_block[b]
        if pair_selection == "extremal":
            theta = cf.get(b)
            if not theta:
                continue
            fees = g["priority_fee"].to_numpy()
            hi_pool = np.flatnonzero(fees >= 2 * theta)
            lo_pool = np.flatnonzero(fees <= theta / 2)
            if len(hi_pool) == 0 or len(lo_pool) == 0:
                continue
            idx = np.array([hi_pool[rng.integers(0, len(hi_pool))],
                            lo_pool[rng.integers(0, len(lo_pool))]])
        else:
            idx = rng.integers(0, len(g), size=2)
        if idx[0] == idx[1]:
            continue
        x, y = g.iloc[idx[0]], g.iloc[idx[1]]
        if x["sender"] == y["sender"] or x["priority_fee"] == y["priority_fee"]:
            continue

        hi_is_x = x["priority_fee"] > y["priority_fee"]
        if bool(x["included"]) != bool(y["included"]):
            guess_x = bool(x["included"])
        elif bool(x["included"]) and bool(y["included"]):
            px, py = pos_lookup.get(x["tx_hash"]), pos_lookup.get(y["tx_hash"])
            if px is None or py is None:
                continue
            guess_x = px < py
        else:
            guess_x = bool(rng.integers(0, 2))

        n_ok += 1
        n_correct += int(guess_x == hi_is_x)

    return PairStats(n_ok, n_correct)


# ---------------------------------------------------------------------------
# Loader for the Flashbots mempool-dumpster daily CSV
# ---------------------------------------------------------------------------

MEMPOOL_DUMPSTER_COLUMNS = (
    "timestamp_ms,hash,chain_id,from,to,value,nonce,gas,gas_price,gas_tip_cap,"
    "gas_fee_cap,data_size,data_4bytes,sources,included_at_block_height,"
    "included_block_timestamp_ms,inclusion_delay_ms,tx_type"
)


def load_mempool_dumpster(path: str, base_fee_wei: int | None = None) -> pd.DataFrame:
    """Read a mempool-dumpster daily CSV (or its .zip) into this module's schema.

    Verified column list as of the project README:
        timestamp_ms, hash, chain_id, from, to, value, nonce, gas, gas_price,
        gas_tip_cap, gas_fee_cap, data_size, data_4bytes, sources,
        included_at_block_height, included_block_timestamp_ms, inclusion_delay_ms, tx_type

    priority_fee is taken as gas_tip_cap, the offered max priority fee per gas.
    If base_fee_wei is supplied, the effective priority fee min(tip_cap, fee_cap - base_fee)
    is used instead, which is the quantity a builder actually earns. Prefer the
    effective version block by block when the base fee is available; the offered
    version is a usable approximation and is what a mempool observer sees.

    Two limitations to carry into the write-up, both of which push the measured
    advantage in a direction that must be stated:
      - transactions already included on chain when first seen are discarded by the
        collector, so the candidate set is missing some late arrivals;
      - private order flow never appears in any public mempool feed, so the true
        builder choice set is strictly larger than what is reconstructed here.
    """
    df = pd.read_csv(path)
    missing = {"timestamp_ms", "hash", "from", "gas_tip_cap"} - set(df.columns)
    if missing:
        raise ValueError(f"not a mempool-dumpster CSV, missing columns: {sorted(missing)}")

    out = pd.DataFrame({
        "tx_hash": df["hash"].astype(str),
        "first_seen_ms": df["timestamp_ms"].astype("int64"),
        "sender": df["from"].astype(str),
        "priority_fee": pd.to_numeric(df["gas_tip_cap"], errors="coerce"),
    })
    if base_fee_wei is not None and "gas_fee_cap" in df.columns:
        eff = pd.to_numeric(df["gas_fee_cap"], errors="coerce") - base_fee_wei
        out["priority_fee"] = np.minimum(out["priority_fee"], eff.clip(lower=0))
    return out.dropna(subset=["priority_fee"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(blocks: pd.DataFrame, meta: pd.DataFrame, mempool: pd.DataFrame | None,
        out_dir: str | None = None, seed: int = RNG_DEFAULT_SEED) -> dict:
    rng = np.random.default_rng(seed)

    sim = []
    for b, blk in blocks.groupby("block_number", sort=False):
        sim.append(apply_random_control(blk, draw_R(rng)))
    ctrl = pd.concat(sim, ignore_index=False).sort_index()
    blocks = blocks.assign(control_position=ctrl["control_position"])

    results: dict = {"arms": {}}

    for arm, col in (("status_quo", "position"), ("random_control", "control_position")):
        ps = position_advantage(blocks, position_col=col, seed=seed)
        results["arms"].setdefault(arm, {})["position"] = {
            "accuracy": ps.accuracy, "advantage": ps.advantage, "n_pairs": ps.n_pairs,
        }

    if mempool is not None and len(mempool):
        cands = build_candidates(blocks, meta, mempool)
        delta_tbl, curve = inclusion_advantage(cands)
        clr_tbl, clr_curve = inclusion_advantage_by_clearing(cands, blocks)
        results["inclusion"] = {
            "clearing_ratio": clr_tbl.to_dict(orient="records"),
            "by_stratum": delta_tbl.to_dict(orient="records"),
            "delta_p_max": float(clr_tbl["delta_p"].iloc[0]) if len(clr_tbl) else (
                float(delta_tbl["delta_p"].max()) if len(delta_tbl) else float("nan")),
            "n_candidate_rows": int(len(cands)),
        }
        for arm, col in (("status_quo", "position"), ("random_control", "control_position")):
            for sel in ("extremal", "random"):
                gs = swap_adversary(cands, blocks, position_col=col, seed=seed,
                                    pair_selection=sel)
                results["arms"][arm][f"game_{sel}"] = {
                    "accuracy": gs.accuracy, "advantage": gs.advantage, "n_trials": gs.n_pairs,
                }
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            cands.to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
            curve.to_csv(os.path.join(out_dir, "inclusion_curve_rankbins.csv"), index=False)
            clr_curve.to_csv(os.path.join(out_dir, "inclusion_curve_clearing.csv"), index=False)
            delta_tbl.to_csv(os.path.join(out_dir, "delta_p_by_stratum.csv"), index=False)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    return results


# ---------------------------------------------------------------------------
# Synthetic self-test: the estimators are checked against the theory's three corners
# ---------------------------------------------------------------------------

def _synthetic(n_blocks: int, cap: int, order: str, inclusion: str,
               seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    brows, mrows, metarows = [], [], []
    n_cand = cap * 3

    for b in range(n_blocks):
        fees = rng.lognormal(mean=20.0, sigma=1.0, size=n_cand)
        hashes = ["0x" + hashlib.sha256(f"{b}:{i}".encode()).hexdigest() for i in range(n_cand)]
        senders = ["0x" + hashlib.sha256(f"s{b}:{i}".encode()).hexdigest()[:40] for i in range(n_cand)]
        t = 1_700_000_000_000 + b * 12_000

        for h, s, f in zip(hashes, senders, fees):
            mrows.append({"tx_hash": h, "first_seen_ms": t - 1_000,
                          "sender": s, "priority_fee": int(f)})

        if inclusion == "threshold":
            chosen = np.argsort(-fees)[:cap]
        elif inclusion == "blind":
            chosen = rng.choice(n_cand, size=cap, replace=False)
        else:
            raise ValueError(inclusion)

        sel_h = [hashes[i] for i in chosen]
        sel_s = [senders[i] for i in chosen]
        sel_f = [int(fees[i]) for i in chosen]

        if order == "fee":
            perm = np.argsort(-np.array(sel_f))
        elif order == "random":
            perm = rng.permutation(len(sel_h))
        else:
            raise ValueError(order)

        for pos, k in enumerate(perm):
            brows.append({"block_number": b, "tx_hash": sel_h[k], "position": pos,
                          "sender": sel_s[k], "priority_fee": sel_f[k],
                          "base_fee": 10**9, "gas_used": 21_000,
                          "block_gas_limit": 30_000_000})

        metarows.append({"block_number": b, "timestamp_ms": t, "base_fee": 10**9,
                         "gas_used": 29_000_000, "gas_limit": 30_000_000})

    return pd.DataFrame(brows), pd.DataFrame(metarows), pd.DataFrame(mrows)


def selftest() -> None:
    print("synthetic self-test: estimators against the theory's three corners\n")
    cases = [
        ("fee-sorted order, threshold inclusion", "fee", "threshold",
         "Adv_pos ~ 1, Delta_p ~ 1, game ~ 1"),
        ("random order, threshold inclusion", "random", "threshold",
         "Adv_pos ~ 0, Delta_p ~ 1, game ~ Delta_p  (order-blind corner)"),
        ("random order, fee-blind inclusion", "random", "blind",
         "Adv_pos ~ 0, Delta_p ~ 0, game ~ 0  (Proposition 5 corner)"),
    ]
    ok = True
    for name, order, inclusion, expectation in cases:
        blocks, meta, mempool = _synthetic(n_blocks=40, cap=60, order=order,
                                           inclusion=inclusion)
        res = run(blocks, meta, mempool, out_dir=None)
        sq = res["arms"]["status_quo"]
        d = res["inclusion"]["delta_p_max"]
        print(f"  {name}")
        print(f"    expected            : {expectation}")
        print(f"    Adv_pos (actual)    : {sq['position']['advantage']:.3f}"
              f"   (n={sq['position']['n_pairs']})")
        print(f"    Adv_pos (control)   : {res['arms']['random_control']['position']['advantage']:.3f}")
        print(f"    Delta_p (clearing)  : {d:.3f}")
        print(f"    game extremal  sq   : {sq['game_extremal']['advantage']:.3f}"
              f"   ctrl: {res['arms']['random_control']['game_extremal']['advantage']:.3f}")
        print(f"    game random    sq   : {sq['game_random']['advantage']:.3f}"
              f"   ctrl: {res['arms']['random_control']['game_random']['advantage']:.3f}")

        if order == "fee" and sq["position"]["advantage"] < 0.95:
            ok = False; print("    FAIL: fee-sorted blocks should give position advantage ~1")
        if res["arms"]["random_control"]["position"]["advantage"] > 0.05:
            ok = False; print("    FAIL: the control permutation must be fee-independent")
        if inclusion == "threshold" and d < 0.9:
            ok = False; print("    FAIL: threshold inclusion should give Delta_p ~ 1")
        if inclusion == "threshold" and sq["game_extremal"]["advantage"] < 0.9:
            ok = False; print("    FAIL: extremal swap adversary should win under threshold inclusion")
        if inclusion == "blind" and d > 0.15:
            ok = False; print("    FAIL: fee-blind inclusion should give Delta_p ~ 0")
        if inclusion == "blind" and res["arms"]["random_control"]["game_extremal"]["advantage"] > 0.15:
            ok = False; print("    FAIL: Proposition 5 corner should give ~0 advantage")
        print()

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--blocks"), ap.add_argument("--meta"), ap.add_argument("--mempool")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=RNG_DEFAULT_SEED)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not (a.blocks and a.meta):
        ap.error("--blocks and --meta are required unless --selftest")

    blocks = pd.read_csv(a.blocks)
    meta = pd.read_csv(a.meta)
    mempool = pd.read_csv(a.mempool) if a.mempool else None
    res = run(blocks, meta, mempool, out_dir=a.out, seed=a.seed)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
