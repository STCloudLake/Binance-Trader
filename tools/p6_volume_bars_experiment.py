"""P6-C evidence runner: dollar bars / volume clock + market volume breadth.

Read-only and reproducible.  It reads **only** cached parquet
(``data/market/<SYM>/1m.parquet``), writes nothing except its own JSON evidence
into ``--out`` (default: a temporary directory), and never touches
``data/binance_trader.db``, ``data/models/**`` or ``strategies/**``.

    python tools/p6_volume_bars_experiment.py bars
    python tools/p6_volume_bars_experiment.py gate
    python tools/p6_volume_bars_experiment.py breadth --samples 6 --interval 20
    python tools/p6_volume_bars_experiment.py breadth --from-cache
    python tools/p6_volume_bars_experiment.py all --out %TEMP%/p6c

Three sub-experiments, each printing its own table and writing its own JSON:

``bars``
    For 2–3 symbols build time bars, dollar bars and volume bars from the same
    1-minute prints, then measure skewness, excess kurtosis, Jarque-Bera
    (exact ``chi2(2)`` p-value), lag-1…5 autocorrelation of returns, the ACF of
    ``|returns|`` (volatility clustering), and the causal-construction check.
    Each metric gets an explicit IMPROVED / WORSE / NO-CHANGE verdict for
    dollar-vs-time and volume-vs-time, plus a circular block-bootstrap interval
    on the difference (a difference whose interval excludes 0 is the only shape
    that supports a claim).
``gate``
    The decisive experiment: the *same* ML credibility protocol
    (``core.ml.credibility``, via ``scripts/ml_credibility_measure.py``'s
    functions) on dollar bars versus time bars for one symbol and one period,
    reporting the gate verdict for each.  The gate decides; the script only
    reports.
``breadth``
    Market-level volume breadth from the reachable 24h ticker endpoint
    (``core.market_data.breadth``): total quote volume, up-share, HHI.  Live
    polling is bounded by ``--samples × --interval`` and ``--max-seconds``; the
    observations are appended to the breadth cache so ``--from-cache`` can
    re-measure the same series deterministically.

Determinism: every stochastic step takes a fixed seed (``--seed``, default 0).
The bar and gate numbers are bit-identical across runs; the breadth series is a
*forward* recording, so a rerun of ``breadth`` live polls new data (that is the
honest limit of an endpoint with no history) while ``breadth --from-cache``
reproduces the recorded numbers bit-identically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.market_data import breadth as breadth_mod  # noqa: E402
from core.strategy.volume_bars import (  # noqa: E402
    VolumeClockBuilder, as_of_consistency, bar_returns,
    bootstrap_metric_difference, dollar_bars, excess_kurtosis,
    gap_spanning_returns, historical_bars_unchanged, jarque_bera,
    match_time_interval, notional_threshold, print_gap_times, return_stats,
    returns_excluding_gaps, spacing_seconds, time_bars, volume_bars,
    volume_threshold,
)

DATA = ROOT / "data" / "market"

#: Plan §3 P6-C: the documented USDT universe size from
#: ``docs/overhaul/MARKET_PAGES_API.md`` (a snapshot figure, never a live pin:
#: it is used only as the *expectation* the coverage column is measured against).
DOCUMENTED_USDT_PAIRS = 496


# ── shared helpers ───────────────────────────────────────────────────────

def load_1m(symbol: str, *, tail_days: Optional[float] = None) -> pd.DataFrame:
    """Cached 1-minute prints for one symbol (parquet only; never the DB)."""
    path = DATA / symbol / "1m.parquet"
    if not path.exists():
        raise SystemExit(f"missing cached 1m data: {path}")
    frame = pd.read_parquet(path)
    if tail_days:
        cutoff = frame.index[-1] - pd.Timedelta(days=float(tail_days))
        frame = frame[frame.index >= cutoff]
    return frame


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and (abs(value) >= 1e5 or
                                     (value != 0 and abs(value) < 1e-3)):
        return f"{value:.4g}"
    return f"{value:.{digits}f}"


def _verdict(reference: Optional[float], candidate: Optional[float], *,
             lower_is_better: bool) -> str:
    """IMPROVED / WORSE / NO-CHANGE for ``candidate`` against ``reference``."""
    if reference is None or candidate is None:
        return "NO-DATA"
    if reference == candidate:
        return "NO-CHANGE"
    better = candidate < reference if lower_is_better else candidate > reference
    return "IMPROVED" if better else "WORSE"


def _abs_or_none(value: Optional[float]) -> Optional[float]:
    """``|value|``, or ``None`` — a missing ACF is "no data", never ``0.0``."""
    return None if value is None else abs(float(value))


def build_series(frame: pd.DataFrame, *, target_bars: int,
                 warmup: float) -> tuple[dict[str, pd.DataFrame], dict]:
    """The three samplings of one print frame, sized to the same bar count.

    The dollar/volume thresholds come from the leading ``warmup`` share of the
    prints (never the whole sample — see ``notional_threshold``).  The time
    interval is then solved (``match_time_interval``) so the control series has
    the same number of observations as the dollar series; without that step a
    fatter-tailed short bar would be compared against a thinner-tailed long one
    and the whole table would be an artefact of bar length.  Returns the three
    frames plus the sizing record (targets, realised counts, interval,
    iterations) so the calibration is reported rather than assumed.
    """
    dollar = dollar_bars(frame, notional_threshold(
        frame, target_bars, calibrate_on=warmup))
    reference_bars = len(dollar) if len(dollar) else int(target_bars)
    offset, realised, iterations = match_time_interval(frame, reference_bars)
    series = {
        "time": time_bars(frame, offset),
        "dollar": dollar,
        "volume": volume_bars(frame, volume_threshold(
            frame, target_bars, calibrate_on=warmup)),
    }
    sizing = {
        "dollar_target_bars": int(target_bars),
        "dollar_notional_per_bar": float(notional_threshold(
            frame, target_bars, calibrate_on=warmup)),
        "volume_per_bar": float(volume_threshold(
            frame, target_bars, calibrate_on=warmup)),
        "time_interval": offset,
        "time_solve_iterations": int(iterations),
        "time_realised_bars": int(realised),
    }
    return series, sizing


# ── experiment 1: distributions + causality ──────────────────────────────

def bars_experiment(symbols: list[str], *, target_bars: int, warmup: float,
                    seed: int, n_boot: int, tail_days: Optional[float]) -> dict:
    report: dict[str, Any] = {"target_bars": target_bars, "warmup": warmup,
                              "seed": seed, "symbols": {}}
    for symbol in symbols:
        prints = load_1m(symbol, tail_days=tail_days)
        started = time.time()
        series, sizing = build_series(prints, target_bars=target_bars,
                                      warmup=warmup)
        build_seconds = time.time() - started
        print(f"\n=== {symbol}: {len(prints)} prints -> "
              f"time={len(series['time'])} dollar={len(series['dollar'])} "
              f"volume={len(series['volume'])} bars "
              f"({build_seconds:.2f}s, time grid {sizing['time_interval']} "
              f"in {sizing['time_solve_iterations']} iteration(s)) ===")
        entry: dict[str, Any] = {
            "prints": int(len(prints)),
            "window": [str(prints.index[0]), str(prints.index[-1])],
            "build_seconds": round(build_seconds, 3),
            "bar_counts": {k: int(len(v)) for k, v in series.items()},
            "sizing": sizing,
            "samplings": {},
        }

        # ── causality: append future prints, count revised historical bars ──
        causality: dict[str, Any] = {}
        for name, bars in series.items():
            if name == "time":
                check = as_of_consistency(prints, kind="time",
                                          interval=sizing["time_interval"])
            elif name == "dollar":
                check = as_of_consistency(
                    prints, kind="dollar",
                    notional_per_bar=sizing["dollar_notional_per_bar"])
            else:
                check = as_of_consistency(
                    prints, kind="volume",
                    volume_per_bar=sizing["volume_per_bar"])
            causality[name] = check
        # The streaming form must reproduce the batch form exactly on
        # boundaries and prices (volume differs only by summation order).
        builder = VolumeClockBuilder(
            notional_per_bar=sizing["dollar_notional_per_bar"])
        streamed = builder.push_frame(prints)
        batch = series["dollar"]
        merged = historical_bars_unchanged(streamed, batch)
        volume_gap = None
        if len(streamed) and len(batch) and streamed.index.equals(batch.index):
            left = streamed["volume"].to_numpy(dtype=float)
            right = batch["volume"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(left - right) / np.where(right == 0, np.nan, right)
            volume_gap = float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0
        entry["causality"] = causality
        entry["streaming_equivalence"] = {
            **merged, "max_volume_rel_diff": volume_gap,
            "boundaries_equal": bool(streamed.index.equals(batch.index)),
            "ohlc_bars_changed": int(max(
                merged["per_column_changed"].get(c, 0)
                for c in ("open", "high", "low", "close")))}
        # Determinism: an independent rebuild must reproduce every bar exactly
        # (same process, same inputs — the "repeated runs bit-identical" row).
        rebuilt, _ = build_series(prints, target_bars=target_bars, warmup=warmup)
        entry["determinism"] = {
            name: bool(rebuilt[name].equals(series[name])) for name in series}

        # ── distribution table ────────────────────────────────────────────
        returns = {name: bar_returns(bars) for name, bars in series.items()}
        stats = {name: return_stats(values) for name, values in returns.items()}
        entry["returns"] = stats
        entry["return_series_sha"] = {
            name: _series_hash(values) for name, values in returns.items()}

        # ── gap sensitivity ───────────────────────────────────────────────
        # The cached print series has multi-week holes (measured on BTCUSDT/1m:
        # 166 476 missing minutes, 24 % of the 697 719-minute span, the two
        # largest 61.8 and 41.6 days).  A time bar resampled across a hole prices
        # a jump that never traded, so the same table is recomputed with the
        # hole-spanning returns removed from *both* series — otherwise a handful
        # of hole returns carries the whole kurtosis comparison.
        holes = print_gap_times(prints)
        gap = {name: gap_spanning_returns(prints, bars)
               for name, bars in series.items()}
        nogap_returns = {name: returns_excluding_gaps(prints, bars)
                         for name, bars in series.items()}
        nogap_stats = {name: return_stats(values)
                       for name, values in nogap_returns.items()}
        entry["print_holes"] = {
            "n": int(len(holes)),
            "missing_minutes": _missing_minutes(prints),
            "span_minutes": float(
                (prints.index[-1] - prints.index[0]).total_seconds() / 60.0),
            "largest_seconds": (float(spacing_seconds(
                prints).max()) if len(prints) > 1 else 0.0),
        }
        entry["gap_returns"] = {
            name: {"n_flagged": int(mask.sum()),
                   "share_of_returns": (float(mask.mean()) if len(mask) else None)}
            for name, mask in gap.items()}
        entry["returns_excluding_gaps"] = nogap_stats

        print(f"{'sampling':8} {'bars':>6} {'skew':>9} {'exkurt':>9} "
              f"{'JB':>10} {'acf1':>8} {'mean|acf|':>9} {'vol-clust':>9}")
        for name in ("time", "dollar", "volume"):
            row = stats[name]
            print(f"{name:8} {row['n']:>6} {_fmt(row['skew']):>9} "
                  f"{_fmt(row['excess_kurtosis']):>9} "
                  f"{_fmt(row['jarque_bera']):>10} "
                  f"{_fmt((row['acf_returns'] or {}).get(1)):>8} "
                  f"{_fmt(row['mean_abs_acf']):>9} "
                  f"{_fmt((row['acf_abs_returns'] or {}).get(1)):>9}")
        print(f"{'sampling':8} {'n':>6} {'skew':>9} {'exkurt':>9} {'JB':>10} "
              f"{'gap_returns':>11}  (same, gap-spanning returns excluded)")
        for name in ("time", "dollar", "volume"):
            row = nogap_stats[name]
            print(f"{name:8} {row['n']:>6} {_fmt(row['skew']):>9} "
                  f"{_fmt(row['excess_kurtosis']):>9} "
                  f"{_fmt(row['jarque_bera']):>10} "
                  f"{entry['gap_returns'][name]['n_flagged']:>11}")

        verdicts: dict[str, Any] = {}
        for candidate in ("dollar", "volume"):
            verdicts[candidate] = {
                "skew": _verdict(stats["time"]["skew"], stats[candidate]["skew"],
                                 lower_is_better=True),
                "excess_kurtosis": _verdict(
                    stats["time"]["excess_kurtosis"],
                    stats[candidate]["excess_kurtosis"], lower_is_better=True),
                "jarque_bera": _verdict(
                    stats["time"]["jarque_bera"],
                    stats[candidate]["jarque_bera"], lower_is_better=True),
                "acf1_returns": _verdict(
                    _abs_or_none((stats["time"]["acf_returns"] or {}).get(1)),
                    _abs_or_none((stats[candidate]["acf_returns"] or {}).get(1)),
                    lower_is_better=True),
                "mean_abs_acf": _verdict(
                    stats["time"]["mean_abs_acf"],
                    stats[candidate]["mean_abs_acf"], lower_is_better=True),
                "excess_kurtosis_excluding_gaps": _verdict(
                    nogap_stats["time"]["excess_kurtosis"],
                    nogap_stats[candidate]["excess_kurtosis"],
                    lower_is_better=True),
                "jarque_bera_excluding_gaps": _verdict(
                    nogap_stats["time"]["jarque_bera"],
                    nogap_stats[candidate]["jarque_bera"],
                    lower_is_better=True),
                "kurtosis_diff_ci": bootstrap_metric_difference(
                    returns[candidate], returns["time"], excess_kurtosis,
                    n_boot=n_boot, seed=seed),
                "jb_diff_ci": bootstrap_metric_difference(
                    returns[candidate], returns["time"], _jb_statistic,
                    n_boot=n_boot, seed=seed),
                "kurtosis_diff_ci_excluding_gaps": bootstrap_metric_difference(
                    nogap_returns[candidate], nogap_returns["time"],
                    excess_kurtosis, n_boot=n_boot, seed=seed),
            }
            # The plan's rule: BOTH the JB statistic and the excess kurtosis must
            # fall, and their bootstrap intervals must exclude zero, before an
            # improvement may be claimed.
            pair = verdicts[candidate]
            pair["claim_supported"] = bool(
                pair["jarque_bera"] == "IMPROVED"
                and pair["excess_kurtosis"] == "IMPROVED"
                and (pair["kurtosis_diff_ci"]["excludes_zero"] or False)
                and (pair["jb_diff_ci"]["excludes_zero"] or False))
            pair["claim_supported_excluding_gaps"] = bool(
                pair["jarque_bera_excluding_gaps"] == "IMPROVED"
                and pair["excess_kurtosis_excluding_gaps"] == "IMPROVED"
                and (pair["kurtosis_diff_ci_excluding_gaps"]["excludes_zero"]
                     or False))
        entry["verdicts"] = verdicts
        report["symbols"][symbol] = entry
    return report


def _missing_minutes(frame: pd.DataFrame) -> float:
    """Minutes absent from the span a print frame covers (holes included)."""
    span = (frame.index[-1] - frame.index[0]).total_seconds() / 60.0
    return float(span - (len(frame) - 1))


def _jb_statistic(values) -> Optional[float]:
    out = jarque_bera(values)["statistic"]
    return None if out is None else float(out)


def _series_hash(values: pd.Series) -> str:
    payload = np.ascontiguousarray(values.to_numpy(dtype=float))
    return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


# ── experiment 2: the ML credibility gate ────────────────────────────────

def gate_experiment(symbol: str, *, target_bars: int, warmup: float,
                    forward: int, threshold: float,
                    tail_days: Optional[float]) -> dict:
    """The same 39-feature credibility protocol on dollar bars vs time bars."""
    from app.config import Config
    from core.ml.credibility import (cost_pct_for, evaluate_model_oos,
                                     gate_from_evaluation)
    from core.ml.features import (FEATURE_NAMES, REQUIRED_INDICATORS,
                                  compute_features, feature_schema_hash,
                                  near_constant_columns)
    from core.ml.labels import CLASS_DOWN, CLASS_UP, create_three_class_label
    from core.ml.trainer import default_binary_factory
    from core.strategy.indicators import compute_all

    config = Config.load("sim")
    cost_pct = cost_pct_for(config, symbol=symbol)
    prints = load_1m(symbol, tail_days=tail_days)
    series, sizing = build_series(prints, target_bars=target_bars,
                                  warmup=warmup)

    report: dict[str, Any] = {
        "symbol": symbol, "cost_pct": cost_pct, "forward": forward,
        "threshold": threshold, "target_bars": target_bars,
        "warmup": warmup,
        "feature_contract_declared": len(FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(),
        "sizing": sizing,
        "gate_config": {
            "auc_min": float(getattr(config, "ml_gate_auc_min", 0.55)),
            "min_net_expectancy": float(
                getattr(config, "ml_gate_net_expectancy_min", 0.0)),
            "min_trades": int(getattr(config, "ml_gate_min_trades", 100)),
            "min_t_stat": float(getattr(config, "ml_gate_min_t_stat", 2.0)),
            "min_psr": float(getattr(config, "ml_gate_min_psr", 0.95)),
            "min_oos": int(getattr(config, "ml_min_oos_rows", 100)),
        },
        "samplings": {},
    }
    for name in ("time", "dollar"):
        bars = series[name]
        indicators = compute_all(bars.copy(), REQUIRED_INDICATORS)
        features = compute_features(indicators)
        close = bars["close"].astype(float)
        fwd = (close.shift(-forward) - close) / close
        three = create_three_class_label(bars, forward_periods=forward,
                                         threshold=threshold, cost_pct=cost_pct)
        binary = pd.Series(np.nan, index=bars.index)
        binary[three == CLASS_UP] = 1.0
        binary[three == CLASS_DOWN] = 0.0
        keep = features.index.intersection(binary.dropna().index)
        result = evaluate_model_oos(
            features.loc[keep], binary.loc[keep], fwd.loc[keep], n_splits=5,
            label_span=forward, cost_pct=cost_pct,
            model_factory=default_binary_factory(), calibrate="isotonic",
            min_trades=int(report["gate_config"]["min_trades"]))
        entry: dict[str, Any] = {
            "bars": int(len(bars)),
            "feature_columns": int(features.shape[1]),
            "labelled_rows": int(len(keep)),
            "window": [str(bars.index[0]), str(bars.index[-1])],
            "near_constant": near_constant_columns(features.loc[keep]),
        }
        if "error" in result:
            entry["error"] = result["error"]
            entry["gate"] = gate_from_evaluation(result)
        else:
            gate = gate_from_evaluation(
                result, auc_min=report["gate_config"]["auc_min"],
                min_trades=report["gate_config"]["min_trades"],
                min_t_stat=report["gate_config"]["min_t_stat"],
                min_psr=report["gate_config"]["min_psr"],
                min_net_expectancy=report["gate_config"]["min_net_expectancy"],
                min_oos=report["gate_config"]["min_oos"])
            entry.update({
                "gate": gate,
                "auc": result["metrics"]["auc"],
                "accuracy": result["metrics"]["accuracy"],
                "majority_accuracy": result["metrics"]["majority_accuracy"],
                "brier": result["metrics"]["brier"],
                "base_rate": result["base_rate"],
                "net_expectancy_oos": result["net_expectancy_oos"],
                "n_trades_oos": result["n_trades_oos"],
                "t_stat_oos": result["t_stat_oos"],
                "psr_oos": result["psr_oos"],
                "n_oos": result["n_oos"],
                "n_splits": result["n_splits"],
            })
        report["samplings"][name] = entry

    print(f"\n=== gate: {symbol} (cost {cost_pct:.4f}%, forward {forward} bars, "
          f"threshold {threshold:.3%}, {report['feature_schema_hash']} v"
          f"{report['feature_contract_declared']}-name contract) ===")
    print(f"{'sampling':8} {'bars':>6} {'feat':>5} {'OOS':>6} {'AUC':>7} "
          f"{'net%':>9} {'trades':>7} {'t':>7} {'PSR':>6}  gate")
    for name in ("time", "dollar"):
        row = report["samplings"][name]
        gate = row["gate"]
        print(f"{name:8} {row['bars']:>6} {row.get('feature_columns', 0):>5} "
              f"{row.get('n_oos', 0):>6} "
              f"{_fmt(row.get('auc')):>7} "
              f"{_fmt((row.get('net_expectancy_oos') or 0.0) * 100):>9} "
              f"{row.get('n_trades_oos', 0):>7} "
              f"{_fmt(row.get('t_stat_oos')):>7} "
              f"{_fmt(row.get('psr_oos')):>6}  "
              f"{'PASS' if gate.get('allowed') else 'FAIL'}")
        if not gate.get("allowed"):
            print(f"         reason: {gate.get('reason')}")
    time_gate = report["samplings"]["time"]["gate"]
    dollar_gate = report["samplings"]["dollar"]["gate"]
    report["comparison"] = {
        "time_allowed": bool(time_gate.get("allowed")),
        "dollar_allowed": bool(dollar_gate.get("allowed")),
        "gate_improved": bool(dollar_gate.get("allowed")
                              and not time_gate.get("allowed")),
        "auc_delta": (report["samplings"]["dollar"].get("auc", 0.5)
                      - report["samplings"]["time"].get("auc", 0.5)),
        "net_delta_pct": ((report["samplings"]["dollar"].get(
            "net_expectancy_oos", 0.0)
            - report["samplings"]["time"].get("net_expectancy_oos", 0.0)) * 100.0),
    }
    verdict = ("dollar bars passed the gate and time bars did not"
               if report["comparison"]["gate_improved"] else
               "dollar bars did NOT improve the gate verdict")
    report["comparison"]["verdict"] = verdict
    print(f"verdict: {verdict} "
          f"(AUC delta {report['comparison']['auc_delta']:+.4f}, "
          f"net delta {report['comparison']['net_delta_pct']:+.4f} pp)")
    return report


# ── experiment 3: market breadth ─────────────────────────────────────────

def breadth_experiment(*, samples: int, interval_s: float, max_seconds: float,
                       data_dir: str, host: str, from_cache: bool,
                       expected_pairs: int, min_quote_volume: float) -> dict:
    cache = breadth_mod.BreadthCache(breadth_mod.default_cache_path(data_dir))
    report: dict[str, Any] = {
        "host": host, "from_cache": from_cache,
        "policy": {
            "ticker24h_ttl_s": breadth_mod.TICKER24H_TTL_S,
            "max_stale_ms": breadth_mod.MAX_STALE_MS,
            "fetch_timeout_s": breadth_mod.FETCH_TIMEOUT_S,
            "request_attempts": breadth_mod.REQUEST_ATTEMPTS,
            "cache_path": str(cache.path),
        },
        "documented_expected_pairs": int(expected_pairs),
    }

    # (a) unreachable-endpoint behaviour — deterministic, no network.
    def _explode():
        raise OSError("simulated: endpoint unreachable")

    unreachable = breadth_mod.fetch_breadth(host=host, fetcher=_explode)
    offline_cache = breadth_mod.BreadthCache(
        Path(tempfile.mkdtemp(prefix="p6c_breadth_")) / "empty.jsonl")
    offline_obs, offline_status = offline_cache.refresh(
        host=host, fetcher=_explode)
    report["unreachable"] = {
        "fetch_returns": None if unreachable is None else "value",
        "fetch_is_none": unreachable is None,
        "cache_status": offline_status,
        "cache_value_is_none": offline_obs is None,
    }
    print(f"unreachable endpoint -> fetch_breadth={unreachable!r}, "
          f"cache status={offline_status!r}, value={offline_obs!r}")

    if not from_cache:
        started = time.time()
        taken = 0
        for index in range(int(samples)):
            if time.time() - started > float(max_seconds):
                print(f"[breadth] time budget {max_seconds:.0f}s reached after "
                      f"{taken} sample(s)")
                break
            observation, status = cache.refresh(
                host=host, expected_pair_count=int(expected_pairs),
                min_quote_volume=float(min_quote_volume), force=True)
            taken += 1
            if observation is None:
                print(f"[breadth] sample {taken}: {status}")
            else:
                print(f"[breadth] sample {taken}: {status} as_of={observation.as_of_ms} "
                      f"pairs={observation.pair_count} usable={observation.usable_count} "
                      f"cov={_fmt(observation.coverage)} "
                      f"total={_fmt(observation.total_quote_volume)} "
                      f"up={_fmt(observation.up_share)} hhi={_fmt(observation.hhi)} "
                      f"fetch={_fmt(observation.fetch_ms, 1)}s")
            if index + 1 < int(samples):
                time.sleep(float(interval_s))

    observations = cache.load()
    report["n_observations"] = len(observations)
    report["observations"] = [obs.to_dict() for obs in observations]
    report["availability"] = breadth_mod.availability_report(observations)
    if observations:
        first, last = observations[0], observations[-1]
        print(f"\n=== breadth: {len(observations)} observation(s), "
              f"{first.as_of_ms} -> {last.as_of_ms} "
              f"({(last.as_of_ms - first.as_of_ms) / 1000.0:.0f}s of wall clock) ===")
        print(f"{'field':20} {'n':>4} {'variance':>14} {'acf1':>9} "
              f"{'acf1(diff)':>11} {'min':>14} {'max':>14}")
        for name in ("total_quote_volume", "up_share", "hhi"):
            row = report["availability"][name]
            print(f"{name:20} {row['n']:>4} {_fmt(row['variance']):>14} "
                  f"{_fmt(row['acf1']):>9} {_fmt(row['acf1_differenced']):>11} "
                  f"{_fmt(row['min']):>14} {_fmt(row['max']):>14}")
        coverage = report["availability"]["coverage"]
        print(f"coverage: min={_fmt(coverage['min'])} mean={_fmt(coverage['mean'])} "
              f"over n={coverage['n']}")
        print(f"non-degenerate (|acf1| < 0.99): "
              f"{report['availability']['non_degenerate_acf_lt_0_99']}")
        print(f"variance > 0: {report['availability']['variance_gt_zero']}")
        print(f"causal labels: {report['availability']['causal_labels']['causal']}")
        report["ttl_documented"] = True
    else:
        print("\nno breadth observations recorded (endpoint unreachable or "
              "empty cache) — nothing to measure, and no value is fabricated")
    return report


# ── driver ───────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command",
                        choices=["bars", "gate", "breadth", "all"],
                        help="which sub-experiment to run")
    parser.add_argument("--symbols", nargs="*",
                        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--target-bars", type=int, default=4000)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--forward", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--tail-days", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--host", default=breadth_mod.DEFAULT_HOST)
    parser.add_argument("--expected-pairs", type=int,
                        default=DOCUMENTED_USDT_PAIRS)
    parser.add_argument("--min-quote-volume", type=float, default=0.0)
    parser.add_argument("--from-cache", action="store_true",
                        help="breadth: re-measure the recorded series, no network")
    parser.add_argument("--out", default=None,
                        help="directory for the evidence JSON (default: temp)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(
        tempfile.mkdtemp(prefix="p6c_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "generated_by": "tools/p6_volume_bars_experiment.py",
        "phase": "P6-C",
        "argv": sys.argv[1:],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if args.command in ("bars", "all"):
        evidence["bars"] = bars_experiment(
            args.symbols, target_bars=args.target_bars, warmup=args.warmup,
            seed=args.seed, n_boot=args.n_boot, tail_days=args.tail_days)
    if args.command in ("gate", "all"):
        evidence["gate"] = gate_experiment(
            args.symbol, target_bars=args.target_bars, warmup=args.warmup,
            forward=args.forward, threshold=args.threshold,
            tail_days=args.tail_days)
    if args.command in ("breadth", "all"):
        evidence["breadth"] = breadth_experiment(
            samples=args.samples, interval_s=args.interval,
            max_seconds=args.max_seconds, data_dir=args.data_dir,
            host=args.host, from_cache=args.from_cache,
            expected_pairs=args.expected_pairs,
            min_quote_volume=args.min_quote_volume)

    evidence["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = out_dir / f"p6c_volume_bars_breadth_{args.command}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, default=str, sort_keys=True)
    print(f"\nEvidence written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
