"""Phase P2 evidence script — ML credibility before/after on real cached data.

Read-only: reads ``data/market/<SYMBOL>/<TF>.parquet`` and writes its JSON
evidence into the directory given by ``--out`` (default: a temp dir).  It never
touches ``data/binance_trader.db`` or ``data/models/``.

    python scripts/ml_credibility_measure.py                 # real BTC/ETH 1h
    python scripts/ml_credibility_measure.py --intervals 1h 1m
    python scripts/ml_credibility_measure.py --synthetic     # gate pass/fail proof

Reported per symbol: base rate, majority-class accuracy, model accuracy, AUC,
Brier, log loss and net-of-cost expectancy, for the **legacy** pipeline
(chronological 80/20, ``±0.5 %`` binary label, "no-move" treated as DOWN, fixed
0.5 threshold) and for the **P2** pipeline (purged K-fold with
``embargo = label horizon``, sample-uniqueness weights, isotonic calibration,
cost-aware threshold).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ml.calibration import ProbabilityCalibrator, reliability_curve, signed_score
from core.ml.credibility import (
    cost_aware_threshold, cost_pct_for, credibility_gate, evaluate_model_oos,
    gate_from_evaluation, ml_accuracy_neutral_abstention,
)
from core.ml.evaluation import (
    average_label_overlap, binary_metrics, overlap_count, purged_kfold_splits,
    sample_uniqueness_weights,
)
from core.ml.features import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, REQUIRED_INDICATORS, compute_features,
    feature_schema_hash, near_constant_columns,
)
from core.ml.labels import class_distribution, create_three_class_label, CLASS_DOWN, CLASS_UP
from core.ml.trainer import default_binary_factory
from core.strategy.indicators import compute_all

DATA = ROOT / "data" / "market"


def load(symbol: str, interval: str, tail: int | None = None) -> pd.DataFrame:
    path = DATA / symbol / f"{interval}.parquet"
    df = pd.read_parquet(path)
    if tail and len(df) > tail:
        df = df.iloc[-tail:].copy()
    return df


def prepare(symbol: str, interval: str, tail: int | None = None):
    df = load(symbol, interval, tail)
    ind = compute_all(df.copy(), REQUIRED_INDICATORS)
    X = compute_features(ind)
    return df, ind, X


# ── legacy pipeline ──────────────────────────────────────────────────────

def legacy_evaluation(symbol: str, interval: str, X: pd.DataFrame, df: pd.DataFrame,
                      cost_pct: float, forward: int = 4, threshold: float = 0.005) -> dict:
    """Old behaviour: chronological 80/20, noise bars folded into class 0."""
    close = df["close"].astype(float)
    fwd = (close.shift(-forward) - close) / close
    label = pd.Series(np.nan, index=df.index)
    label[fwd >= threshold] = 1.0
    label[fwd <= -threshold] = 0.0
    # The deployed pipeline can only emit two classes, so a no-move bar had to be
    # forced into one of them; `_predict_ml` treated everything <= 0.5 as bearish.
    label[label.isna()] = 0.0
    keep = X.index.intersection(fwd.dropna().index)
    Xv, yv, rv = X.loc[keep], label.loc[keep], fwd.loc[keep]

    split = int(len(Xv) * 0.8)
    factory = default_binary_factory()
    model = factory(Xv.iloc[:split], yv.iloc[:split].astype(float))
    if model is None:
        return {"error": "no model"}
    p_te = model.predict_proba(Xv.iloc[split:])[:, 1]
    y_te = yv.iloc[split:].values
    r_te = rv.iloc[split:].values
    metrics = binary_metrics(y_te, p_te)
    metrics["cost_pct"] = cost_pct
    # Legacy decision: >0.5 = long, else short (engine `_predict_ml` semantics).
    take_long = p_te > 0.5
    exp = cost_aware_threshold(y_te, p_te, r_te, cost_pct=cost_pct)
    with np.errstate(invalid="ignore"):
        net_long = float(np.mean(r_te[take_long] - cost_pct / 100.0)) if take_long.any() else 0.0
    metrics["net_expectancy_at_0.5"] = net_long
    metrics["reliability"] = reliability_curve(y_te, p_te)
    metrics["n_train"] = int(split)
    metrics["n_test"] = int(len(Xv) - split)
    metrics["thresholds"] = {k: v for k, v in exp.items() if k != "curve"}
    return metrics


# ── P2 pipeline ──────────────────────────────────────────────────────────

def _holdout_calibrated(result: dict) -> np.ndarray:
    """Honest calibration check: fit on the first half of the OOS rows,
    transform the second half.  (The pooled calibrated curve is fitted and
    measured on the same rows, so it is monotone *by construction* — ECE 0 —
    and must not be quoted as evidence of generalisation.)"""
    p = np.asarray(result["p_raw_oos"], dtype=float)
    y = np.asarray(result["y_oos"], dtype=float)
    cut = len(p) // 2
    if cut < 60:
        return p
    cal = ProbabilityCalibrator("isotonic").fit(p[:cut], y[:cut])
    out = p.copy()
    out[cut:] = cal.transform(p[cut:])
    return out


def p2_evaluation(symbol: str, interval: str, X: pd.DataFrame, ind: pd.DataFrame,
                  df: pd.DataFrame, cost_pct: float, forward: int = 4,
                  threshold: float = 0.005, cost_multiple: float = 0.0,
                  config=None) -> dict:
    close = df["close"].astype(float)
    fwd = (close.shift(-forward) - close) / close
    three = create_three_class_label(df, forward_periods=forward, threshold=threshold,
                                     cost_pct=cost_pct, cost_multiple=cost_multiple)
    binary = pd.Series(np.nan, index=df.index)
    binary[three == CLASS_UP] = 1.0
    binary[three == CLASS_DOWN] = 0.0
    keep = X.index.intersection(binary.dropna().index)
    Xv, yv, rv = X.loc[keep], binary.loc[keep], fwd.loc[keep]

    gate_cfg = {}
    if config is not None:
        gate_cfg = {
            "auc_min": float(getattr(config, "ml_gate_auc_min", 0.55)),
            "min_net_expectancy": float(getattr(config, "ml_gate_net_expectancy_min", 0.0)),
            "min_trades": int(getattr(config, "ml_gate_min_trades", 100)),
            "min_t_stat": float(getattr(config, "ml_gate_min_t_stat", 2.0)),
            # Audit F3/F5: the PSR floor was previously not passed at all, so this
            # measurement script gated on the hard-coded default while
            # `ml.gate_min_psr` was loaded and unread.
            "min_psr": float(getattr(config, "ml_gate_min_psr", 0.95)),
            "min_oos": int(getattr(config, "ml_min_oos_rows", 100)),
        }
    calibration = str(getattr(config, "ml_calibration", "isotonic") or "isotonic") \
        if config is not None else "isotonic"
    result = evaluate_model_oos(
        Xv, yv, rv, n_splits=5, label_span=forward, cost_pct=cost_pct,
        model_factory=default_binary_factory(), calibrate=calibration,
        min_trades=int(gate_cfg.get("min_trades", 100)))
    if "error" in result:
        return result
    gate = gate_from_evaluation(result, **gate_cfg)

    # Reliability of the UNcalibrated probabilities on the same OOS rows (mean
    # per-fold AUC): this is what the audit measured as "inverted".
    raw_auc = float(np.mean([f["auc_uncalibrated"] for f in result["folds"]])) \
        if result.get("folds") else result["metrics"]["auc"]

    return {
        "n": result["n"],
        "n_oos": result["n_oos"],
        "n_splits": result["n_splits"],
        "folds": result["folds"],
        "base_rate": float(yv.mean()),
        "flat_share": class_distribution(three)["timeout_share"],
        "cost_multiple": cost_multiple,
        "flat_share_all_bars": float((three == 2).sum()) / max(len(three.dropna()), 1),
        "metrics": result["metrics"],
        "thresholds": {k: v for k, v in result["thresholds"].items() if k != "curve"},
        "thresholds_oos": result["thresholds_oos"],
        "threshold_curve": result["thresholds"]["curve"],
        # The honest outer numbers (audit P2 #1) — the gate consumes these.
        "net_expectancy_oos": result["net_expectancy_oos"],
        "n_trades_oos": result["n_trades_oos"],
        "t_stat_oos": result["t_stat_oos"],
        "psr_oos": result["psr_oos"],
        "ci_low_oos": result["ci_low_oos"],
        "ci_high_oos": result["ci_high_oos"],
        "net_expectancy_pooled_optimistic": result["net_expectancy_pooled"],
        "coverage_oos": (result["n_trades_oos"] / result["n_oos"]
                         if result["n_oos"] else 0.0),
        "reliability_calibrated": reliability_curve(result["y_oos"], result["p_oos"]),
        "reliability_calibrated_oos": reliability_curve(
            result["y_oos"], _holdout_calibrated(result)),
        "reliability_raw": reliability_curve(result["y_oos"], result["p_raw_oos"]),
        "metrics_uncalibrated": result["metrics_uncalibrated"],
        "auc_uncalibrated_mean_fold": raw_auc,
        "gate": gate,
        "calibrator": result["calibrator"],
        "calibration_n_fit_sum": int(sum(f["calibrator_n_fit"] for f in result["folds"])),
        "calibration_stream_rows_sum": int(sum(f["n_cal"] for f in result["folds"])),
        "feature_schema_hash": feature_schema_hash(),
        "n_features": int(Xv.shape[1]),
        "near_constant": near_constant_columns(Xv),
    }


def purge_proof(X: pd.DataFrame, forward: int = 4, n_splits: int = 5) -> dict:
    """Row counts + how many overlapping labels the old split leaked."""
    n = len(X)
    splits = purged_kfold_splits(n, n_splits, label_span=forward)
    per_fold = []
    total_purged = 0
    for train_idx, test_idx, purged in splits:
        leaked = overlap_count(test_idx, forward, n)
        total_purged += purged
        per_fold.append({
            "train": int(len(train_idx)), "test": int(len(test_idx)),
            "purged_plus_embargo": int(purged),
            "old_split_overlapping_rows": int(leaked),
        })
    # The single chronological 80/20 split the old trainer used.
    split = int(n * 0.8)
    old_leak = int((np.minimum(np.arange(split) + forward, n - 1) >= split).sum())
    weights = sample_uniqueness_weights(n, forward)
    return {
        "n": n, "label_span": forward, "folds": per_fold,
        "total_purged_plus_embargo": int(total_purged),
        "old_chronological_split_overlap_rows": old_leak,
        "old_chronological_train_rows": split,
        "average_label_window_overlap": average_label_overlap(n, forward),
        "uniqueness_weight_sum": float(weights.sum()),
        "uniqueness_weight_min": float(weights.min()),
        "uniqueness_weight_max": float(weights.max()),
    }


# ── synthetic gate proof ─────────────────────────────────────────────────

def synthetic_case(kind: str, seed: int = 42, n: int = 3000) -> dict:
    """A model with real signal must pass the gate; pure noise must fail it."""
    rng = np.random.default_rng(seed)
    strength = {"signal": 0.55, "noise": 0.0, "short": 0.55}[kind]
    count = 300 if kind == "short" else n
    x = rng.normal(size=(count, 3))
    logit = strength * x[:, 0] * 2.5
    p_true = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(count) < p_true).astype(float)
    if kind == "noise":
        y = (rng.random(count) < 0.5).astype(float)
    # Costs are subtracted from every taken trade, so an informative signal
    # still has to clear the fee to pass.
    fwd = np.where(y > 0.5, 0.006, -0.006) + rng.normal(scale=0.001, size=count)
    X = pd.DataFrame(x, columns=["a", "b", "c"])
    return {"X": X, "y": pd.Series(y), "fwd": pd.Series(fwd)}


def calibration_holdout_proof(seed: int = 11, n: int = 6000) -> dict:
    """Reliability curve before (inverted) vs after (monotone) on held-out data.

    Two distortions are measured, both on the second half only after fitting on
    the first half:

    ``inverted``   logit flipped in sign — the audit's signature failure
                   (predicted 0.91 → realised 0.22).  Isotonic cannot recover a
                   signal from a reversed ordering, so this case documents that
                   a *retrain* (not a calibration) is the required fix; its
                   reliability curve is the negative-slope one.
    ``overconfident``  rank preserved but the spread compressed toward 0.5 —
                   the case calibration is meant to fix: ECE must fall and the
                   curve must become monotone (Spearman ≥ 0.8).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p_good = 1.0 / (1.0 + np.exp(-1.6 * z))
    y = (rng.random(n) < p_good).astype(float)
    logit = np.log(np.clip(p_good, 1e-6, 1 - 1e-6) / (1 - np.clip(p_good, 1e-6, 1 - 1e-6)))
    cut = n // 2

    out = {}
    cases = {
        "inverted": 1.0 / (1.0 + np.exp(0.8 * logit + 0.3)),
        "overconfident": 0.5 + 0.6 * (p_good - 0.5),
    }
    for name, p_bad in cases.items():
        cal = ProbabilityCalibrator("isotonic").fit(p_bad[:cut], y[:cut])
        p_cal = cal.transform(p_bad[cut:])
        out[name] = {
            "before": reliability_curve(y[cut:], p_bad[cut:]),
            "after": reliability_curve(y[cut:], p_cal),
            "auc_before": binary_metrics(y[cut:], p_bad[cut:])["auc"],
            "auc_after": binary_metrics(y[cut:], p_cal)["auc"],
        }
    return out


def synthetic_gate_proof(cost_pct: float) -> dict:
    out = {}
    for kind in ("signal", "noise", "short"):
        case = synthetic_case(kind)
        res = evaluate_model_oos(
            case["X"], case["y"], case["fwd"], n_splits=5, label_span=2,
            cost_pct=cost_pct, model_factory=default_binary_factory(),
            calibrate="isotonic", min_train=50)
        if "error" in res:
            out[kind] = {"error": res["error"],
                         "gate": {"allowed": False, "enabled": False,
                                  "reason": res["error"]}}
            continue
        gate = gate_from_evaluation(res)
        out[kind] = {
            "auc": gate["auc"], "net_expectancy": gate["net_expectancy"],
            "n_oos": gate["n_oos"], "allowed": gate["allowed"],
            "enabled": gate["enabled"], "reason": gate["reason"],
        }
    return out


def diag_proof() -> dict:
    """engine.py diagnostic: neutral-as-0.5 (old) vs neutral-as-abstention (new)."""
    rng = np.random.default_rng(7)
    conf = np.concatenate([rng.uniform(0.3, 0.5, 400), np.full(300, 0.5),
                           rng.uniform(0.5, 0.7, 300)])
    ret = rng.normal(scale=0.01, size=len(conf))
    old_total = old_correct = 0
    for c, r in zip(conf, ret):
        if abs(r) >= 0.005:
            old_total += 1
            if (r >= 0.005 and c >= 0.5) or (r <= -0.005 and c < 0.5):
                old_correct += 1
    return {
        "old_engine_style": {
            "n_total": old_total,
            "accuracy_pct": round(old_correct / old_total * 100, 1) if old_total else 0.0,
            "neutral_band_counted_as_bearish": int((conf == 0.5).sum()),
        },
        "new_helper": ml_accuracy_neutral_abstention(list(zip(conf, ret)), [], threshold=0.005),
        "signed_score_examples": {
            "p=0.60,base=0.45": round(signed_score(0.60, 0.45), 4),
            "p=0.40,base=0.45": round(signed_score(0.40, 0.45), 4),
            "p=0.45,base=0.45": round(signed_score(0.45, 0.45), 4),
            "p=0.35,base=0.30": round(signed_score(0.35, 0.30), 4),
        },
    }


def selection_bias_proof(seed: int = 42, n: int = 2500) -> dict:
    """Item 1 evidence: pooled calibration+selection vs the nested protocol.

    The audit's half-split experiment on real data gave OOS net **−0.075 %
    (BTC) / −0.342 % (ETH)** where the pooled pipeline reported +0.358 % /
    +0.058 %.  This runs both protocols on a synthetic matrix that carries a
    *conditional* edge (the top of the probability range is profitable) so the
    difference is visible deterministically:

    ``pooled``
        one calibrator + one threshold, both fitted on the rows they are scored
        on (the pre-fix behaviour),
    ``nested``
        per-fold calibrator on the fold's calibration stream and the threshold
        selected inside the fold — :func:`evaluate_model_oos`'s protocol.
    """
    from core.ml.credibility import evaluate_model_oos as _eval
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    p_true = 1.0 / (1.0 + np.exp(-(1.2 * x[:, 0] - 0.4 * x[:, 1])))
    y = (rng.random(n) < p_true).astype(float)
    # Edge only above the base rate: a threshold search is required to see it.
    fwd = np.where(p_true > 0.55, 0.010, -0.004) + rng.normal(scale=0.002, size=n)
    X = pd.DataFrame(x, columns=["a", "b", "c"])
    res = _eval(X, pd.Series(y), pd.Series(fwd), n_splits=5, label_span=2,
                cost_pct=0.25, model_factory=default_binary_factory(),
                calibrate="isotonic", min_train=50, min_trades=50)
    if "error" in res:
        return {"error": res["error"]}
    from core.ml.credibility import credibility_gate, gate_from_evaluation
    return {
        "pooled_optimistic_net": res["net_expectancy_pooled"],
        "pooled_threshold": (res["thresholds"] or {}).get("threshold"),
        "pooled_n_taken": (res["thresholds"] or {}).get("n_taken"),
        "nested_outer_net": res["net_expectancy_oos"],
        "nested_n_trades": res["n_trades_oos"],
        "nested_t_stat": res["t_stat_oos"],
        "nested_ci": [res["ci_low_oos"], res["ci_high_oos"]],
        "calibration_n_fit_sum": int(sum(f["calibrator_n_fit"] for f in res["folds"])),
        "calibration_stream_rows_sum": int(sum(f["n_cal"] for f in res["folds"])),
        "gate": gate_from_evaluation(res, min_trades=50),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--intervals", nargs="*", default=["1h"])
    ap.add_argument("--tail", type=int, default=None, help="use only the last N bars")
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cost-multiple", type=float, default=0.0,
                    help="effective label threshold = max(threshold, k * cost)")
    ap.add_argument("--config", default=None, help="path to config dir (default: repo)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="ml_p2_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.config import Config
    config = Config.load("sim")
    report: dict = {
        "feature_schema_version": int(FEATURE_SCHEMA_VERSION),
        "feature_schema_hash": feature_schema_hash(),
        "n_features": len(FEATURE_NAMES),
        "measurements": {},
    }

    for symbol in args.symbols:
        for interval in args.intervals:
            cost_pct = cost_pct_for(config, symbol=symbol)
            df, ind, X = prepare(symbol, interval, args.tail)
            key = f"{symbol}_{interval}"
            print(f"[{key}] rows={len(df)} features={X.shape[1]} cost={cost_pct:.4f}%")
            legacy = legacy_evaluation(symbol, interval, X, df, cost_pct)
            p2 = p2_evaluation(symbol, interval, X, ind, df, cost_pct,
                               cost_multiple=0.0, config=config)
            p2_cost = p2_evaluation(symbol, interval, X, ind, df, cost_pct,
                                    cost_multiple=4.0, config=config)
            report["measurements"][key] = {
                "rows": int(len(df)),
                "cost_pct": cost_pct,
                "legacy": legacy,
                "p2": p2,
                "p2_cost_aware_threshold": p2_cost,
                "purge_proof": purge_proof(X),
            }
            if "metrics" in legacy and "metrics" in p2:
                lm, pm = legacy["metrics"], p2["metrics"]
                print(f"  legacy: acc={lm['accuracy']:.4f} maj={lm['majority_accuracy']:.4f} "
                      f"auc={lm['auc']:.4f} brier={lm['brier']:.4f} "
                      f"logloss={lm['log_loss']:.4f} net@0.5={lm['net_expectancy_at_0.5']*100:.4f}%")
                print(f"  p2    : acc={pm['accuracy']:.4f} maj={pm['majority_accuracy']:.4f} "
                      f"auc={pm['auc']:.4f} brier={pm['brier']:.4f} "
                      f"logloss={pm['log_loss']:.4f} "
                      f"OOS_net={p2['net_expectancy_oos']*100:.4f}% "
                      f"n_trades={p2['n_trades_oos']} t={p2['t_stat_oos']:.2f} "
                      f"CI=[{p2['ci_low_oos']*100:.4f}%, {p2['ci_high_oos']*100:.4f}%] "
                      f"gate={'PASS' if p2['gate']['allowed'] else 'FAIL'}")
                print(f"          pooled(optimistic) net={p2['net_expectancy_pooled_optimistic']*100:.4f}% "
                      f"n_taken={(p2['thresholds'] or {}).get('n_taken')} "
                      f"| cal_stream_rows={p2['calibration_stream_rows_sum']} "
                      f"cal_n_fit={p2['calibration_n_fit_sum']}")
                print(f"          gate_reason={p2['gate']['reason']}")
                print(f"          reliability raw  monotone={p2['reliability_raw']['monotone']} "
                      f"slope={p2['reliability_raw']['slope']:.3f}")
                print(f"          reliability cal  monotone={p2['reliability_calibrated']['monotone']} "
                      f"slope={p2['reliability_calibrated']['slope']:.3f}")

    report["diagnostic_proof"] = diag_proof()
    report["calibration_holdout_proof"] = calibration_holdout_proof()
    report["selection_bias_proof"] = selection_bias_proof()
    if args.synthetic:
        report["synthetic_gate_proof"] = synthetic_gate_proof(
            cost_pct_for(config, symbol="BTCUSDT"))

    path = out_dir / "ml_p2_measurements.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nEvidence written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
