"""D1 validation (temporary): causal HMM stability, OOS accuracy, latency."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import core.strategy.regime as R  # noqa: E402


def synth(n=3000, seed=5, cp=(1000, 2000)):
    rng = np.random.default_rng(seed)
    spec = [(0.002, 0.0004), (0.010, -0.0008), (0.002, 0.0004)]
    b = (0, cp[0], cp[1], n)
    rets, truth, trend = [], [], []
    for k, (s, d) in enumerate(spec):
        m = b[k + 1] - b[k]
        rets.append(rng.standard_normal(m) * s + d)
        truth += ["high" if k == 1 else "low"] * m
        trend += ["up" if k != 1 else "down"] * m
    r = np.concatenate(rets)
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1.0}, index=idx), \
        np.asarray(truth), np.asarray(trend)


def main():
    print("=== causal label stability (append future bars) ===")
    _df, truth_all, _tr = synth()
    r = np.log(_df["close"]).diff().to_numpy()
    r = r[np.isfinite(r)]
    truth = truth_all[1:]
    for step in (250, 100, 50):
        f = R.hmm_two_state_causal(r, refit_every=step)
        pred = np.where(f["states"] == 1, "high", "low")
        m = R.detection_metrics(truth, pred, positive="high",
                                change_points=(999, 1999))
        lab = f["state"].astype(str).to_numpy()
        ok = lab != "unknown"
        m2 = R.detection_metrics(truth[ok], pred[ok], positive="high",
                                 change_points=(999, 1999))
        print(f"refit={step}: OOS acc all={m['accuracy']:.4f} lat={m['latency_bars']} "
              f"| labelled={m2['accuracy']:.4f} lat={m2['latency_bars']}")
    for seed in (5, 6, 7):
        df, truth, _ = synth(seed=seed)
        lr = np.log(df["close"]).diff()
        t0 = time.perf_counter()
        full = R.hmm_two_state_causal(lr)
        t1 = time.perf_counter()
        trunc = R.hmm_two_state_causal(lr.iloc[:2000])
        a = full["state"].iloc[:2000].astype(str)
        b = trunc["state"].reindex(a.index).astype(str)
        eq = int((a.to_numpy() == b.to_numpy()).sum())
        print(f"seed{seed} causal fit {t1 - t0:.2f}s refits={full['n_refits']} "
              f"first_label={full['first_label_index']} equal={eq}/{len(a)}")
        # whole-sample (shipped default) for contrast
        f2 = R.hmm_two_state(lr)
        t2 = R.hmm_two_state(lr.iloc[:2000])
        c = f2["state"].iloc[:2000].astype(str)
        d = t2["state"].reindex(c.index).astype(str)
        print(f"       whole-sample mode equal={int((c.to_numpy() == d.to_numpy()).sum())}"
              f"/{len(c)} sigma_full={f2['sigma'][0]:.5f}/{f2['sigma'][1]:.5f} "
              f"sigma_trunc={t2['sigma'][0]:.5f}/{t2['sigma'][1]:.5f}")
        pred = np.where(full["states"] == 1, "high", "low")
        n_pred = len(pred)
        m = R.detection_metrics(truth[1:1 + n_pred], pred, positive="high",
                                change_points=(999, 1999))
        labels = full["state"].astype(str).to_numpy()
        ok = labels != "unknown"
        m2 = R.detection_metrics(truth[1:1 + n_pred][ok], pred[ok], positive="high",
                                 change_points=(999, 1999))
        ins = R.detection_metrics(truth[1:1 + n_pred],
                                  np.where(f2["states"] == 1, "high", "low"),
                                  positive="high", change_points=(999, 1999))
        print(f"       causal OOS acc(all bars)={m['accuracy']:.4f} lat={m['latency_bars']}")
        print(f"       causal OOS acc(labelled)={m2['accuracy']:.4f} lat={m2['latency_bars']} "
              f"n_labelled={int(ok.sum())}")
        print(f"       whole-sample IN-SAMPLE acc={ins['accuracy']:.4f} lat={ins['latency_bars']}")

    print("\n=== gate refusal ===")
    df, _, _ = synth(seed=5)
    tbl_bad = R.classify_regimes(df, with_hmm=True, causal_hmm=False)
    tbl_ok = R.classify_regimes(df, with_hmm=True, causal_hmm=True)
    print("attrs bad:", tbl_bad.attrs, "attrs ok:", tbl_ok.attrs)
    gate = R.default_gate(enabled=True)
    try:
        print(R.gate_regimes(gate, tbl_bad, "breakout"))
    except R.NonCausalRegimeError as e:
        print("REFUSED non-causal table:", e)
    print("causal table ->", R.gate_regimes(gate, tbl_ok, "breakout"))
    off = R.default_gate(enabled=False)
    print("gate off ->", R.gate_regimes(off, tbl_bad, "breakout"))

    print("\n=== default resolution (gating off keeps whole-sample mode) ===")
    f_default = R.hmm_two_state(np.log(df["close"]).diff())
    print("causal flag with gating off:", f_default["causal"])
    R.REGIME_GATING_ENABLED = True
    try:
        f_on = R.hmm_two_state(np.log(df["close"]).diff())
        print("causal flag with gating on:", f_on["causal"])
    finally:
        R.REGIME_GATING_ENABLED = False


if __name__ == "__main__":
    main()
