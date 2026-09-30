"""Feature engineering for ML models.

The canonical feature set is :data:`FEATURE_NAMES` — a single list shared by the
live predictor, the trainers and the backtest path.  Phase P2 removed the
"live trains on 40, backtest uses 47" fork (the deployed pkl even carried
``n_features_in_=40`` while the backtest matrix had 47 columns) by making this
module the only place a feature list is defined, and by having
:func:`compute_features` refuse to return a short matrix.

Feature groups:
- Price momentum (6): multi-period returns + acceleration
- Volatility (4): rolling std + volatility regime
- Volume (4): volume changes + trend
- Price position (5): distance from EMAs, BB position, BB width trend
- Trend/indicator (6): RSI, MACD, ADX + their changes
- Microstructure (2): intra-bar position, high-low range
- Sequence (3): consecutive direction, return distribution shape
- Market structure (5): rolling Hurst, swing distances, swing range, reversals
- Fractional memory (3): fractional-differenced returns + their volatility

Phase P2 audit finding (item 5): **10 of the old 40 columns were literal
constants on the production path** because ``REQUIRED_INDICATORS`` only computed
rsi/macd/bollinger/adx, so ``hurst``/``hurst_signal``/``roll_hurst_20`` fell back
to 0.5, the swing features to 0.0 and the fractional ones to 0.0 while the model
happily assigned them importance.  The indicator config below now computes every
input these features need (``core.strategy.indicators.compute_all`` supports
``atr``/``hurst``/``swing_points``/``frac_diff``), and
:func:`assert_no_constant_columns` turns a regression into a test failure.
"""

import hashlib
import json

import pandas as pd
import numpy as np


# ── Canonical feature list (39 features) ────────────────────────────────

DEFAULT_FEATURES: list[str] = [
    # Price momentum
    "ret_1", "ret_5", "ret_10", "ret_20",
    "acceleration_5", "momentum_ratio",
    # Volatility
    "vol_5", "vol_10", "vol_20", "vol_regime",
    # Volume
    "volume_ratio", "vol_chg_5", "vol_chg_20", "vol_trend",
    # Price position
    "pos_20", "ema20_dist", "ema50_dist",
    "bb_position", "bb_width_ratio",
    # Trend / indicator
    "rsi", "macd_histogram", "bollinger_width", "adx",
    "rsi_momentum", "macd_accel",
    # Microstructure
    "close_position", "high_low_range",
    # Sequence / distribution
    "consecutive_dir", "ret_skew_20", "ret_kurt_20",
    # ── Market structure (Phase 4c) ──
    # NOTE: bare "hurst" was dropped — it is perfectly collinear with
    # `roll_hurst_20` (its own 20-bar mean), so keeping both added a duplicate
    # column without adding information.
    "hurst_signal",
    "dist_to_swing_high", "dist_to_swing_low",
    "swing_range_pct",
    "swing_reversal_count_50",
    "roll_hurst_20",
    # ── Fractional memory ──
    "frac_ret_5", "frac_ret_10", "frac_vol_10",
]

#: The one and only feature contract.  Trainers persist this list in the model
#: metadata and refuse to score a matrix whose columns differ (item 5).
FEATURE_NAMES: tuple[str, ...] = tuple(DEFAULT_FEATURES)

#: Columns ``compute_features`` always produces even when not selected — they are
#: cheap, genuinely varying extras, deliberately NOT part of the contract.
OPTIONAL_FEATURES: tuple[str, ...] = (
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "volume_price_corr_20", "adx_trend_strength", "ema_slope_20",
)

# Indicator configs needed to compute the indicator-derived features above.
# `swing_points`/`frac_diff` are what make the market-structure block
# non-constant; `atr` feeds the volatility-scaled triple barrier.
#
# `hurst` is deliberately NOT in this dict (audit P2 #7): the legacy indicator
# (`core.strategy.indicators._compute_hurst_indicator`) costs O(n · lookback²)
# with a 10-lag R/S regression per bar — measured 2.096 s for 8 844 bars against
# 0.0029 s before it was added to the set (730×) — and `hurst_signal` was just
# its own rolling mean, i.e. the same number twice.  The features below compute a
# **bounded** R/S Hurst instead: a 60-bar window on a 4-lag grid, refreshed every
# `_HURST_STRIDE` bars and forward-filled (see `_rolling_hurst_bounded`).
REQUIRED_INDICATORS: dict[str, dict] = {
    "rsi": {"period": 14, "source": "close"},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "stddev": 2},
    "adx": {"period": 14},
    "atr": {"period": 14},
    "swing_points": {"lookback": 5},
    "frac_diff": {"d": 0.4, "threshold": 0.001},
}

#: Rolling-Hurst budget: window length, lags used per window and the refresh
#: stride.  ``60``-bar windows on ``4`` lags is the cheapest setting that still
#: yields a varying series on 8 800 real 1h bars (measured std ≈ 0.076).
HURST_LOOKBACK = 60
HURST_LAGS = (4, 6, 9, 12)
_HURST_STRIDE = 4


#: Which DataFrame columns each entry of :data:`REQUIRED_INDICATORS` produces.
#: Used by :func:`compute_features` to refuse a raw-OHLCV input instead of
#: silently emitting constant columns.
INDICATOR_COLUMNS: dict[str, tuple[str, ...]] = {
    "rsi": ("rsi",),
    "macd": ("macd_histogram",),
    "bollinger": ("bollinger_upper", "bollinger_lower"),
    "adx": ("adx",),
    "atr": ("atr",),
    "swing_points": ("swing_high", "swing_low"),
    "frac_diff": ("frac_close",),
}


def _indicator_presence(df: pd.DataFrame, name: str) -> list[str]:
    """The columns of indicator *name* that are actually present in *df*."""
    return [c for c in INDICATOR_COLUMNS.get(name, ()) if c in df.columns]


#: Rows required before an indicator-derived feature matrix is meaningful.
#: ``compute_features`` raises below this and the live predictor
#: (``MLPredictor._on_kline``) refuses to score below it — before P2 the
#: predictor guarded at 100 rows while the feature code needed 200, so the
#: 100–199-row band produced an all-zero/ffilled matrix (audit P2 #10).
MIN_FEATURE_ROWS = HURST_LOOKBACK + 140


class FeatureContractError(RuntimeError):
    """Raised when a feature matrix violates the canonical contract."""


def feature_schema_hash(feature_names: list[str] | tuple[str, ...] | None = None) -> str:
    """Stable hash of a feature list — stored in model metadata."""
    names = list(FEATURE_NAMES if feature_names is None else feature_names)
    return hashlib.sha1(json.dumps(names).encode("utf-8")).hexdigest()[:12]


def missing_feature_columns(result: pd.DataFrame,
                            feature_list: list[str] | tuple[str, ...] | None = None
                            ) -> list[str]:
    """Canonical (or requested) features absent from *result*."""
    wanted = list(FEATURE_NAMES if feature_list is None else feature_list)
    return [f for f in wanted if f not in result.columns]


def validate_feature_matrix(
    result: pd.DataFrame,
    feature_list: list[str] | tuple[str, ...] | None = None,
    *,
    require_exact: bool = False,
    near_constant: bool = False,
    tolerance: float = 1e-9,
) -> None:
    """Fail loudly when a feature matrix does not match the contract.

    ``require_exact`` additionally rejects extra columns (the live/backtest
    parity assertion).  ``near_constant`` rejects columns whose standard
    deviation (or whole range) is below *tolerance* — the test that would have
    caught the 10 constant columns.
    """
    wanted = list(FEATURE_NAMES if feature_list is None else feature_list)
    missing = [f for f in wanted if f not in result.columns]
    if missing:
        raise FeatureContractError(
            f"feature matrix is missing {len(missing)} required column(s): "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"(have {len(result.columns)}, expected {len(wanted)})")
    if require_exact:
        extra = [c for c in result.columns if c not in wanted]
        if extra:
            raise FeatureContractError(f"feature matrix has unexpected column(s): {extra[:8]}")
        if list(result.columns) != wanted:
            raise FeatureContractError(
                "feature matrix column ORDER differs from the contract "
                "(a LightGBM/XGBoost model scores positionally)")
    if near_constant:
        bad = near_constant_columns(result[wanted], tolerance=tolerance)
        if bad:
            raise FeatureContractError(f"near-constant feature column(s): {bad}")


def near_constant_columns(
    result: pd.DataFrame,
    tolerance: float = 1e-9,
    min_unique: int = 2,
) -> list[str]:
    """Columns that are constant (or effectively constant) across all rows."""
    bad: list[str] = []
    for col in result.columns:
        series = pd.to_numeric(result[col], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) == 0:
            bad.append(str(col))
            continue
        if finite.nunique(dropna=True) < int(min_unique):
            bad.append(str(col))
            continue
        if float(finite.std(ddof=0)) <= float(tolerance):
            bad.append(str(col))
            continue
        if float(finite.max() - finite.min()) <= float(tolerance):
            bad.append(str(col))
    return bad


def _rs_hurst_lags(returns: np.ndarray, lags: tuple[int, ...] = HURST_LAGS) -> float:
    """R/S Hurst exponent for one window of log-returns on a fixed lag grid."""
    n = len(returns)
    if n < 20:
        return 0.5
    usable = [lag for lag in lags if 4 <= lag <= n // 2]
    if len(usable) < 2:
        return 0.5
    rs_values: list[float] = []
    used: list[int] = []
    for lag in usable:
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        chunks = returns[: n_chunks * lag].reshape(n_chunks, lag).astype(np.float64)
        mean = chunks.mean(axis=1, keepdims=True)
        cum_dev = (chunks - mean).cumsum(axis=1)
        rng = cum_dev.max(axis=1) - cum_dev.min(axis=1)
        scale = chunks.std(axis=1, ddof=1) + 1e-12
        rs_values.append(float((rng / scale).mean()))
        used.append(lag)
    if len(rs_values) < 2:
        return 0.5
    slope = float(np.polyfit(np.log(used), np.log(rs_values), 1)[0])
    return max(0.0, min(1.0, slope))


def _rolling_hurst_bounded(
    close: pd.Series,
    *,
    lookback: int = HURST_LOOKBACK,
    stride: int = _HURST_STRIDE,
) -> pd.Series:
    """Bounded rolling R/S Hurst of log-returns (audit P2 #7).

    Same estimator as ``core.strategy.indicators._compute_hurst_indicator`` but
    with a bounded window/lag grid and refreshed every ``stride`` bars, then
    forward-filled: live inference cannot afford a 10-lag regression per bar
    (measured 2.096 s per call on 8 844 bars).  Bars before the first full window
    are NaN; the caller fills them like every other feature.
    """
    prices = close.to_numpy(dtype=float)
    n = len(prices)
    out = np.full(n, np.nan)
    if n <= lookback + 1:
        return pd.Series(out, index=close.index)
    log_prices = np.log(np.maximum(prices, 1e-12))
    step = max(int(stride), 1)
    for i in range(lookback, n, step):
        out[i] = _rs_hurst_lags(np.diff(log_prices[i - lookback: i + 1]))
    series = pd.Series(out, index=close.index)
    # Forward-fill the stride gaps (each value is a stale-but-knowable window).
    return series.ffill()


def build_features(df: pd.DataFrame, feature_list: list[str] | None = None) -> pd.DataFrame:
    """Extract named feature columns from a DataFrame that already has them.

    This is the simple path — call it when indicators are pre-computed on *df*.
    """
    if feature_list is None:
        feature_list = DEFAULT_FEATURES
    available = [f for f in feature_list if f in df.columns]
    if not available:
        return pd.DataFrame(index=df.index)
    result = df[available].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.ffill().fillna(0)
    return result


def compute_features(df: pd.DataFrame,
                     feature_list: list[str] | None = None,
                     *,
                     validate: bool = True) -> pd.DataFrame:
    """Compute the canonical feature set from an indicator-bearing DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least 'open','high','low','close','volume' plus every
        indicator column produced by
        ``core.strategy.indicators.compute_all(df, REQUIRED_INDICATORS)``.
        Passing a subset silently produced 10 constant columns before P2, so a
        missing indicator input now raises :class:`FeatureContractError`.
    feature_list : list[str] | None
        Subset of :data:`FEATURE_NAMES` to return; ``None`` = the full contract.
        ``[]`` is treated as ``None`` for backward compatibility with callers
        that pass an empty list meaning "everything".
    validate : bool
        Assert the resulting matrix matches the requested contract and has no
        near-constant column (default True — this is the fail-loudly path).

    Returns
    -------
    pd.DataFrame
        Feature matrix with the same index as *df*.
    """
    if feature_list is not None and len(feature_list) == 0:
        feature_list = None
    # The documented data requirement: below this every rolling window (and the
    # 60-bar Hurst) is NaN→ffill→0, i.e. a matrix of constants.  The live
    # predictor guards on the same constant so the two cannot disagree.
    if len(df) < MIN_FEATURE_ROWS:
        raise FeatureContractError(
            f"compute_features needs at least {MIN_FEATURE_ROWS} rows "
            f"(got {len(df)}); shorter windows make every rolling feature "
            f"constant (audit P2 #10)")
    # None → the canonical contract (NOT "every column computed"): the old
    # `feature_list=None ⇒ all computed columns` behaviour is exactly how the
    # live path (40) and the backtest path (47) drifted apart (item 5).
    effective_list = list(FEATURE_NAMES if feature_list is None else feature_list)

    required_inputs = [
        col for name in REQUIRED_INDICATORS for col in INDICATOR_COLUMNS.get(name, ())
    ]
    missing_inputs = [c for c in required_inputs if c not in df.columns]
    if missing_inputs:
        raise FeatureContractError(
            "compute_features needs the indicator columns produced by "
            f"compute_all(df, REQUIRED_INDICATORS); missing: {missing_inputs}. "
            "Feeding it raw OHLCV silently yields constant features "
            "(this was audit finding P2-5).")

    result = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    vol = df.get("volume", pd.Series(0, index=df.index)).astype(float)

    # ── Price momentum ──────────────────────────────────────────────
    result["ret_1"] = close.pct_change(1)
    result["ret_5"] = close.pct_change(5)
    result["ret_10"] = close.pct_change(10)
    result["ret_20"] = close.pct_change(20)
    # Acceleration: how much the recent trend is speeding up / slowing down
    result["acceleration_5"] = result["ret_1"] - result["ret_5"].shift(5)
    # Momentum ratio: short-term vs medium-term — >1 = accelerating trend
    result["momentum_ratio"] = (result["ret_5"].abs() + 1e-9) / (
        result["ret_20"].abs() + 1e-9)

    # ── Volatility ──────────────────────────────────────────────────
    ret = result["ret_1"]
    result["vol_5"] = ret.rolling(5).std()
    result["vol_10"] = ret.rolling(10).std()
    result["vol_20"] = ret.rolling(20).std()
    # Volatility regime: expanding (>1) or contracting (<1)
    result["vol_regime"] = (result["vol_5"] + 1e-9) / (result["vol_20"] + 1e-9)

    # ── Volume ──────────────────────────────────────────────────────
    vol_sma_5 = vol.rolling(5).mean()
    vol_sma_20 = vol.rolling(20).mean()
    result["volume_ratio"] = vol / (vol_sma_20 + 1e-9)
    result["vol_chg_5"] = vol.pct_change(5)
    result["vol_chg_20"] = vol.pct_change(20)
    result["vol_trend"] = (vol_sma_5 + 1e-9) / (vol_sma_20 + 1e-9)

    # ── Price position ──────────────────────────────────────────────
    result["pos_20"] = (close - close.rolling(20).min()) / (
        close.rolling(20).max() - close.rolling(20).min() + 1e-9)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    result["ema20_dist"] = (close - ema20) / (ema20 + 1e-9)
    result["ema50_dist"] = (close - ema50) / (ema50 + 1e-9)

    # BB position (normalised 0-1 within bands)
    if "bollinger_upper" in df.columns and "bollinger_lower" in df.columns:
        bb_upper = df["bollinger_upper"]
        bb_lower = df["bollinger_lower"]
        result["bb_position"] = (close - bb_lower) / (
            bb_upper - bb_lower + 1e-9)
        # BB width relative to its own 20-period SMA
        if "bollinger_width" in df.columns:
            bw_sma20 = df["bollinger_width"].rolling(20).mean()
            result["bb_width_ratio"] = (df["bollinger_width"] + 1e-9) / (bw_sma20 + 1e-9)
        else:
            result["bb_width_ratio"] = 1.0
    else:
        result["bb_position"] = 0.5
        result["bb_width_ratio"] = 1.0

    # ── Trend / indicator ───────────────────────────────────────────
    for col in ["rsi", "macd_histogram", "bollinger_width", "adx"]:
        if col in df.columns:
            result[col] = df[col]
    if "rsi" in result.columns:
        result["rsi_momentum"] = result["rsi"] - result["rsi"].shift(5)
    else:
        result["rsi_momentum"] = 0.0
    if "macd_histogram" in result.columns:
        result["macd_accel"] = (result["macd_histogram"] -
                                result["macd_histogram"].shift(3))
    else:
        result["macd_accel"] = 0.0

    # ── Microstructure ──────────────────────────────────────────────
    high = df.get("high", close)
    low = df.get("low", close)
    result["close_position"] = (close - low) / (high - low + 1e-9)
    result["high_low_range"] = (high - low) / (close + 1e-9)

    # ── Sequence / distribution ─────────────────────────────────────
    # Consecutive directional bars (approximate — uses close vs prev close)
    direction = np.sign(close.diff(1))
    result["consecutive_dir"] = (
        direction.groupby((direction != direction.shift(1)).cumsum())
        .cumcount()
        .astype(float)
    )
    # Distribution shape of recent returns
    result["ret_skew_20"] = ret.rolling(20).skew()
    result["ret_kurt_20"] = ret.rolling(20).kurt()

    # ── Temporal features (cyclical encoding for intraday patterns) ──
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour.astype(float)
        days = df.index.dayofweek.astype(float)
        result["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        result["day_of_week_sin"] = np.sin(2 * np.pi * days / 7)
        result["day_of_week_cos"] = np.cos(2 * np.pi * days / 7)
    else:
        for col in ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]:
            result[col] = 0.0

    # ── Volume-quality ───────────────────────────────────────────────
    price_dir = np.sign(close.diff(1))
    vol_dir = np.sign(vol.diff(1))
    result["volume_price_corr_20"] = (
        price_dir.rolling(20).corr(vol_dir)
    ).fillna(0)

    # ── Trend quality ───────────────────────────────────────────────
    result["adx_trend_strength"] = df.get("adx", pd.Series(0, index=df.index)) / 100.0
    result["ema_slope_20"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-9)

    # ── Market structure (Phase 4c) ──────────────────────────────────
    # The raw `hurst` level is computed here with the bounded estimator (audit
    # P2 #7); `roll_hurst_20` is its own 20-bar mean, which is why the bare
    # `hurst` column is not part of the shipped contract.
    result["hurst"] = _rolling_hurst_bounded(close)
    result["roll_hurst_20"] = result["hurst"].rolling(20).mean()
    # `hurst_signal` is the *regime change* — short-term Hurst minus its 60-bar
    # mean — not a second copy of the same rolling mean.  Before P2 it was
    # `hurst.rolling(lookback).mean()`, i.e. perfectly collinear with the
    # 100-bar mean of the series it shipped next to.  Both come from the same
    # bounded series, so the extra column costs nothing.
    result["hurst_signal"] = result["hurst"] - result["hurst"].rolling(
        HURST_LOOKBACK).mean()

    # Swing point distances (requires swing_points indicator)
    result["dist_to_swing_high"] = pd.to_numeric(df["dist_to_high_pct"], errors="coerce")
    result["dist_to_swing_low"] = pd.to_numeric(df["dist_to_low_pct"], errors="coerce")
    result["swing_range_pct"] = pd.to_numeric(df["swing_range_pct"], errors="coerce")
    # Reversal activity: bars where the confirmed swing HIGH moved in the last
    # 50 bars (the old expression re-added the swing type regardless of change,
    # so it was indistinguishable from a 50-period count).
    result["swing_reversal_count_50"] = (
        df["swing_high"].diff().fillna(0.0).ne(0.0).astype(float)
        .rolling(50).sum())

    # Fractional-differenced returns (requires frac_diff indicator).  The
    # fractional series is a *level* built from long-memory weights, whose
    # magnitude depends on the price scale; pct_change of a small residual is
    # numerically unstable, so the contract uses the fractional change
    # normalised by entry price (a stationarised, scale-free measure).
    fd = pd.to_numeric(df["frac_close"], errors="coerce")
    for horizon in (5, 10):
        change = (fd - fd.shift(horizon)) / (close.abs() + 1e-12)
        result[f"frac_ret_{horizon}"] = change.clip(-0.5, 0.5)
    result["frac_vol_10"] = result["frac_ret_5"].rolling(10).std()

    # ── Cleanup ─────────────────────────────────────────────────────
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.ffill().fillna(0)

    # Subset to the contract (or the caller's explicit subset).  `effective_list`
    # is used instead of `feature_list` so the None case is the contract too.
    available = [f for f in effective_list if f in result.columns]
    result = result[available]

    if validate:
        validate_feature_matrix(result, effective_list,
                                require_exact=(feature_list is None),
                                near_constant=True)

    return result


# ── Label helpers ────────────────────────────────────────────────────────


def create_binary_label(df: pd.DataFrame, forward_periods: int = 4,
                         threshold: float = 0.005) -> pd.Series:
    """Binary label with noise filter.

    Returns 1 if price rises >= threshold, 0 if falls >= threshold,
    NaN for insignificant (noise) moves.

    .. deprecated:: P2
        This target **drops the no-move regime** (BTCUSDT 1h: 59.5 % of bars;
        1m: 84.7 %), which is exactly why the live model could not beat the
        majority class on the bars it was asked to decide.  New code should use
        :func:`core.ml.labels.create_three_class_label` (keeps ``flat`` as a
        class and abstains on it) or
        :func:`core.ml.labels.create_triple_barrier_label_vol`.  Kept only for
        the backtest engine, which is out of this phase's write scope.
    """
    future_close = df["close"].shift(-forward_periods)
    return_pct = (future_close - df["close"]) / df["close"]
    labels = pd.Series(np.nan, index=df.index)
    labels[return_pct >= threshold] = 1.0
    labels[return_pct <= -threshold] = 0.0
    return labels.astype("Int64")


def create_regression_label(df: pd.DataFrame,
                             forward_periods: int = 4) -> pd.Series:
    """Continuous label: forward return percentage."""
    future_close = df["close"].shift(-forward_periods)
    return (future_close - df["close"]) / df["close"]


# ── Triple Barrier Labels ────────────────────────────────────────────────

def create_triple_barrier_label(
    df: pd.DataFrame,
    forward_periods: int = 24,
    upper_pct: float = 0.02,
    lower_pct: float = 0.02,
    timeout_label: float | None = None,
    *,
    vol_scaled: bool = True,
    atr_period: int = 14,
    atr_multiple: float = 1.5,
    min_pct: float = 0.004,
    max_pct: float = 0.06,
    max_rows: int | None = None,
) -> pd.Series:
    """Path-aware label using the Triple Barrier Method.

    Instead of asking "did price go up at time T+N?", we ask "which
    barrier did price hit FIRST within the next N periods?"

    This respects the *path* — a price that rallies 3% then crashes
    5% hits the upper barrier first (label=1), even though the endpoint
    return is negative. Binary labels would incorrectly label this 0.

    Phase P2: barriers are **volatility-scaled by default** (``vol_scaled=True``
    → :func:`core.ml.labels.create_triple_barrier_label_vol`, width
    ``atr_multiple × ATR / close`` clamped to ``[min_pct, max_pct]``).  The
    fixed ``24 × 2 %`` version made 44.9 % of all labels timeouts on real data,
    which is a class, not a signal.  Pass ``vol_scaled=False`` for the legacy
    fixed-width behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'high', 'low', 'close' columns.
    forward_periods : int
        Maximum number of periods to look forward — the time barrier.  Set it
        to the strategy's real maximum holding period (bar count).
    upper_pct, lower_pct : float
        Fixed barriers (used only when ``vol_scaled=False``).
    timeout_label : float | None
        Value for samples where no barrier is hit (NaN = filtered out).
    vol_scaled : bool
        Use ATR-scaled barriers (default True).
    atr_multiple, min_pct, max_pct : float
        Barrier width controls for the volatility-scaled mode.
    max_rows : int | None
        Label only the most recent ``max_rows`` bars (research convenience).

    Returns
    -------
    pd.Series with same index as df:
        1  = upper barrier hit first (bullish)
        0  = lower barrier hit first (bearish)
        NaN or timeout_label = neither barrier hit within window
    """
    if vol_scaled:
        from core.ml.labels import create_triple_barrier_label_vol
        return create_triple_barrier_label_vol(
            df, forward_periods=forward_periods, atr_period=atr_period,
            atr_multiple=atr_multiple, min_pct=min_pct, max_pct=max_pct,
            timeout_label=timeout_label, max_rows=max_rows)

    n = len(df)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    labels = np.full(n, np.nan)

    for i in range(n - forward_periods):
        entry = close[i]
        upper = entry * (1.0 + upper_pct)
        lower = entry * (1.0 - lower_pct)

        hit = np.nan
        for j in range(1, forward_periods + 1):
            idx = i + j
            if idx >= n:
                break
            if high[idx] >= upper:
                hit = 1.0  # upper hit first
                break
            if low[idx] <= lower:
                hit = 0.0  # lower hit first
                break
        labels[i] = hit

    result = pd.Series(labels, index=df.index)
    if timeout_label is not None:
        # Genuine timeouts inside the sample become the timeout class; the last
        # `forward_periods` rows have no full window and stay NaN (audit P2 #6).
        tail = result.iloc[n - forward_periods:] if forward_periods < n else result
        result = result.fillna(timeout_label)
        result.loc[tail.index] = np.nan
    return result


# ── Volatility & Regime prediction labels ─────────────────────────────────
# Unlike the directional labels above, these predict *volatility state*
# rather than price direction.  Volatility is significantly more
# predictable than returns (GARCH, HAR, etc.), making these labels
# more suitable for ML models in efficient markets.


def create_volatility_label(
    df: pd.DataFrame,
    forward_periods: int = 20,
    comparison_window: int = 20,
) -> pd.Series:
    """Binary label: will future volatility EXPAND (1) or CONTRACT (0)?

    Compares future realised volatility against current realised volatility.
    A label of 1 means volatility is expected to increase — the position
    sizer should respond by reducing exposure.

    Args:
        df: OHLCV DataFrame with at least ``"close"`` column.
        forward_periods: How many bars ahead to measure future volatility.
        comparison_window: Rolling window for volatility estimation.

    Returns:
        pd.Series of 1 (expand) / 0 (contract) / NaN (insufficient data).
    """
    ret = df["close"].pct_change()
    current_vol = ret.rolling(comparison_window).std()
    # Future volatility: the std of returns from t+1 to t+forward_periods+1
    future_vol = (
        ret.shift(-1)
        .rolling(forward_periods)
        .std()
        .shift(-forward_periods + 1)
    )
    labels = pd.Series(np.nan, index=df.index)
    mask = current_vol.notna() & future_vol.notna()
    labels[mask & (future_vol > current_vol)] = 1.0
    labels[mask & (future_vol <= current_vol)] = 0.0
    return labels.astype("Int64")


# ── Hurst exponent estimator (used by labels and indicators) ──────────────

def _estimate_hurst(prices: np.ndarray, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent H via rescaled range (R/S) on log-returns.

    For a time series:  R/S ∝ N^H

    H > 0.55 → trending/persistent
    H ≈ 0.50 → random walk / efficient
    H < 0.45 → mean-reverting

    Uses log returns internally so the caller can pass raw prices.

    Known limitation: classical R/S has a small-sample upward bias
    (Anis-Lloyd correction applicable for N > 100).  For regime
    detection, relative comparison across assets/time-windows is
    more reliable than the absolute value.

    Args:
        prices: 1-D numpy array of prices (NOT returns).
        max_lag: Maximum lag for R/S calculation.

    Returns:
        Estimated Hurst exponent in [0, 1].
    """
    if len(prices) < 60:
        return 0.5

    # Convert to log returns
    log_prices = np.log(np.maximum(prices, 1e-12))
    returns = np.diff(log_prices)
    n = len(returns)

    if n < 50 or max_lag < 4:
        return 0.5

    # Log-spaced lags for efficient sampling
    lags = np.unique(np.logspace(
        np.log10(4), np.log10(min(max_lag, n // 4)), num=12).astype(int))
    if len(lags) < 4:
        return 0.5

    rs_values = []
    for lag in lags:
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        # Non-overlapping chunks
        chunks = returns[: n_chunks * lag].reshape(n_chunks, lag).astype(np.float64)
        # Cumulative deviation from mean within each chunk
        mean = chunks.mean(axis=1, keepdims=True)
        cum_dev = (chunks - mean).cumsum(axis=1)
        R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
        S = chunks.std(axis=1, ddof=1) + 1e-12
        rs_values.append(float((R / S).mean()))

    if len(rs_values) < 4:
        return 0.5

    # log(RS) = H * log(lag) + const  →  H = slope
    log_lags = np.log([lag for lag, _ in zip(lags, rs_values)])
    log_rs = np.log(rs_values)
    H = float(np.polyfit(log_lags, log_rs, 1)[0])
    return max(0.0, min(1.0, H))
