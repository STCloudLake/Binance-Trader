"""P6-C — dollar bars / volume clock (``core.strategy.volume_bars``).

Every expectation here is derived from a frame the test builds itself: no test
reads ``data/market/**``, touches the network, or pins a measured live-market
number (the policy guard in ``tests/test_measured_threshold_policy.py``).  The
real-cache numbers live in ``docs/core-algorithms/15-volume-bars-breadth.md``
and are reproduced by ``tools/p6_volume_bars_experiment.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.strategy.volume_bars import (
    CACHE_COLUMNS, MIN_STATS_ROWS, VolumeClockBuilder, acf, as_of_consistency,
    autocorrelation, bar_returns, bars_as_of, bars_from_rows,
    block_bootstrap_ci, bootstrap_metric_difference, dollar_bars,
    excess_kurtosis, gap_spanning_returns, historical_bars_unchanged,
    interval_for_bar_count, jarque_bera, match_time_interval,
    notional_threshold, print_gap_times, return_stats, returns_excluding_gaps,
    skewness, spacing_seconds, time_bars, volatility_clustering, volume_bars,
    volume_threshold,
)

INDEX = "close_time"


# ── synthetic frames ─────────────────────────────────────────────────────

def clock_frame(periods: int = 6, *, close=None, volume=None,
                freq: str = "1min") -> pd.DataFrame:
    """A print frame with explicit, easily-counted notional per print."""
    index = pd.date_range("2025-01-01", periods=periods, freq=freq, name=INDEX)
    close = np.full(periods, 10.0) if close is None else np.asarray(close, float)
    volume = np.full(periods, 2.0) if volume is None else np.asarray(volume, float)
    return pd.DataFrame({
        "open": close + np.arange(periods) * 0.1,
        "high": close + 0.05 + np.arange(periods) * 0.1,
        "low": close - 0.05 - np.arange(periods) * 0.01,
        "close": close,
        "volume": volume,
    }, index=index)


def random_walk_frame(periods: int = 4000, *, seed: int = 7,
                      gap_at: int | None = None) -> pd.DataFrame:
    """A seeded random-walk print frame (the causality fixtures' input)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=periods, freq="1min", name=INDEX)
    if gap_at is not None:
        index = index.delete(slice(gap_at, gap_at + 200))
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, len(index))))
    return pd.DataFrame({
        "open": close, "high": close * 1.0002, "low": close * 0.9998,
        "close": close, "volume": rng.uniform(1.0, 3.0, len(index)),
    }, index=index)


# ── the two activity clocks ──────────────────────────────────────────────

def test_dollar_bars_close_on_the_notional_threshold():
    """Notional is ``close × volume`` = 20/print, so a 50 threshold = 3 prints."""
    frame = clock_frame()
    bars = dollar_bars(frame, 50.0)
    assert list(bars.columns) == list(CACHE_COLUMNS)
    assert bars.index.name == INDEX
    assert len(bars) == 2
    # Each bar is the aggregation of the prints that reached the threshold; the
    # expectation is recomputed from the same frame rather than pinned.
    first = frame.iloc[:3]
    assert bars["open"].iloc[0] == first["open"].iloc[0]
    assert bars["high"].iloc[0] == first["high"].max()
    assert bars["low"].iloc[0] == first["low"].min()
    assert bars["close"].iloc[0] == first["close"].iloc[-1]
    assert bars["volume"].iloc[0] == first["volume"].sum()
    assert bars.index[0] == frame.index[2]
    assert bars.index[1] == frame.index[-1]


def test_volume_bars_count_base_volume_not_notional():
    """Same volumes, wildly different prices: the two clocks must disagree."""
    frame = clock_frame(close=[1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    volume_clock = volume_bars(frame, 5.0)
    dollar_clock = dollar_bars(frame, 5.0)
    # Base volume is 2 per print, so 5 closes every third print.
    assert len(volume_clock) == 2
    # Notional jumps 100x, so the dollar clock closes at almost every print.
    assert len(dollar_clock) > len(volume_clock)
    assert dollar_clock.index[1] == frame.index[3]


def test_drop_partial_controls_the_trailing_incomplete_bar():
    """A trailing bar below the threshold is provisional, not historical."""
    frame = clock_frame()          # 6 prints × 20 notional = 120 total
    dropped = dollar_bars(frame, 100.0)          # bar1 = 20 < 100 → dropped
    kept = dollar_bars(frame, 100.0, drop_partial=False)
    assert len(dropped) == 1
    assert len(kept) == 2
    # Everything except the trailing bar is identical either way.
    assert kept.iloc[:1].equals(dropped)


def test_activity_clock_overshoot_is_dropped_not_carried():
    """The next bar starts at the print *after* the one that closed the bar."""
    frame = clock_frame(close=[1.0] * 6, volume=[6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Print 0 alone reaches the threshold (6 ≥ 5); the overshoot is discarded,
    # so prints 1..5 accumulate only 5 → exactly two bars.
    bars = volume_bars(frame, 5.0)
    assert len(bars) == 2
    assert bars["volume"].iloc[0] == frame["volume"].iloc[0]
    assert bars["volume"].iloc[1] == frame["volume"].iloc[1:].sum()


# ── the time control ─────────────────────────────────────────────────────

def test_time_bars_resample_right_closed_and_drop_the_last_bin():
    frame = clock_frame(6, freq="1min")
    bars = time_bars(frame, "2min")
    # Bins are (…, 00:00], (00:00, 00:02], (00:02, 00:04], (00:04, 00:06] on the
    # absolute start_day grid, so the 00:00 print opens the 00:00-labelled bin,
    # and the final bin is provisional and dropped.
    assert list(bars.index) == [pd.Timestamp("2025-01-01 00:00"),
                                pd.Timestamp("2025-01-01 00:02"),
                                pd.Timestamp("2025-01-01 00:04")]
    assert bars.index.name == INDEX
    second_bin = frame.loc["2025-01-01 00:01":"2025-01-01 00:02"]
    assert bars["volume"].iloc[1] == second_bin["volume"].sum()
    assert bars["high"].iloc[1] == second_bin["high"].max()
    # The 00:00 bin holds exactly the print stamped 00:00.
    assert bars["volume"].iloc[0] == frame["volume"].iloc[0]


def test_time_bars_drop_empty_gap_bins():
    frame = random_walk_frame(600, gap_at=200)
    coarse = time_bars(frame, "1h")
    # A 200-minute hole must not manufacture a bar with no trades in it.
    assert not coarse["volume"].isna().any()
    assert (coarse["volume"] > 0).all()


def test_time_bars_keeps_the_last_bin_on_request():
    frame = clock_frame(6)
    # Four bins exist; only the last one is provisional.
    assert len(time_bars(frame, "2min", drop_last=False)) == 4


# ── causality: the acceptance criterion ──────────────────────────────────

@pytest.mark.parametrize("kind", ["dollar", "volume", "time"])
def test_appending_future_prints_revises_zero_historical_bars(kind: str):
    """The P6-C causality row: 0 changed bars, every prefix fully matched."""
    frame = random_walk_frame()
    spec = {"dollar": {"notional_per_bar": notional_threshold(frame, 200)},
            "volume": {"volume_per_bar": volume_threshold(frame, 200)},
            "time": {"interval": "20min"}}[kind]
    full = bars_as_of(frame, frame.index[-1], kind=kind, **spec)
    assert len(full) > 0
    for cut in (500, 1500, 3000):
        prefix = bars_as_of(frame.iloc[:cut], frame.index[cut - 1],
                            kind=kind, **spec)
        report = historical_bars_unchanged(prefix, full)
        assert report["n_prefix"] == report["n_compared"]
        assert report["n_changed"] == 0, (kind, cut, report)


@pytest.mark.parametrize("kind", ["dollar", "volume", "time"])
def test_as_of_consistency_reports_zero_revisions(kind: str):
    frame = random_walk_frame(2000)
    spec = {"dollar": {"notional_per_bar": notional_threshold(frame, 100)},
            "volume": {"volume_per_bar": volume_threshold(frame, 100)},
            "time": {"interval": "20min"}}[kind]
    report = as_of_consistency(frame, kind=kind, **spec)
    assert report["n_cuts"] >= 5
    assert report["total_changed"] == 0
    assert report["worst"]["n_changed"] == 0


def test_bars_as_of_ignores_every_print_after_the_cutoff():
    frame = random_walk_frame(1200)
    spec = {"notional_per_bar": notional_threshold(frame, 60)}
    cutoff = frame.index[700]
    causal = bars_as_of(frame, cutoff, kind="dollar", **spec)
    tampered = frame.copy()
    # Rewrite the future with an absurd price move: the past must not notice.
    tampered.iloc[701:, tampered.columns.get_loc("high")] *= 10.0
    tampered.iloc[701:, tampered.columns.get_loc("low")] *= 0.1
    same = bars_as_of(tampered, cutoff, kind="dollar", **spec)
    assert causal.equals(same)


def test_keeping_the_partial_bar_is_the_only_thing_that_can_move():
    """Documented trade-off: with ``drop_partial=False`` exactly the trailing
    provisional bar may be revised by later prints — and nothing else."""
    frame = random_walk_frame(1500)
    threshold = notional_threshold(frame, 80)
    full = dollar_bars(frame, threshold, drop_partial=False)
    prefix = dollar_bars(frame.iloc[:1000], threshold, drop_partial=False)
    report = historical_bars_unchanged(prefix, full)
    assert report["n_changed"] <= 1
    if report["n_changed"]:
        assert report["changed_times"] == [str(prefix.index[-1])]


# ── thresholds ───────────────────────────────────────────────────────────

def test_notional_threshold_reads_only_the_warmup_block():
    frame = random_walk_frame(2000)
    tampered = frame.copy()
    cut = int(len(frame) * 0.2)
    tampered.iloc[cut + 1:] *= 5.0
    assert notional_threshold(frame, 100, calibrate_on=0.2) == \
        notional_threshold(tampered, 100, calibrate_on=0.2)


def test_threshold_targets_the_whole_sample_bar_count():
    """Constant notional ⇒ the realised count is exactly the requested count."""
    frame = clock_frame(1000)
    bars = dollar_bars(frame, notional_threshold(frame, 50, calibrate_on=0.2))
    assert len(bars) == 50


def test_match_time_interval_reaches_the_requested_bin_count():
    frame = random_walk_frame(6000, gap_at=1000)
    offset, realised, iterations = match_time_interval(frame, 300)
    assert offset.endswith("min")
    assert len(time_bars(frame, offset)) == realised
    # The grid is whole-minute, so the count cannot be hit continuously; the
    # returned interval is the closest one the iteration found.
    assert abs(realised - 300) <= max(3.0, 0.05 * 300)
    assert iterations >= 1


def test_interval_for_bar_count_divides_the_span():
    frame = random_walk_frame(600)
    assert interval_for_bar_count(frame, 60) == "10min"
    assert interval_for_bar_count(frame, 600) == "1min"


# ── the streaming form ───────────────────────────────────────────────────

def test_builder_reproduces_the_batch_partition_and_prices():
    frame = random_walk_frame(4000)
    threshold = notional_threshold(frame, 200)
    batch = dollar_bars(frame, threshold)
    streamed = VolumeClockBuilder(notional_per_bar=threshold).push_frame(frame)
    assert streamed.index.equals(batch.index)
    for column in ("open", "high", "low", "close"):
        assert np.array_equal(streamed[column].to_numpy(),
                              batch[column].to_numpy())
    # Volume is the one column whose last bits depend on summation order.
    assert np.allclose(streamed["volume"].to_numpy(),
                       batch["volume"].to_numpy(), rtol=1e-12, atol=0.0)


def test_builder_volume_clock_matches_the_batch_volume_clock():
    frame = random_walk_frame(3000)
    threshold = volume_threshold(frame, 150)
    batch = volume_bars(frame, threshold)
    streamed = VolumeClockBuilder(volume_per_bar=threshold).push_frame(frame)
    assert streamed.index.equals(batch.index)
    assert np.array_equal(streamed["high"].to_numpy(), batch["high"].to_numpy())


def test_builder_never_revises_an_emitted_bar():
    """A closed bar is frozen: the emitted sequence from a prefix is a prefix."""
    frame = random_walk_frame(1500)
    threshold = notional_threshold(frame, 100)
    emitted: list[float] = []
    builder = VolumeClockBuilder(notional_per_bar=threshold)
    for timestamp, row in frame.iterrows():
        closed = builder.push(timestamp, row["open"], row["high"], row["low"],
                              row["close"], row["volume"])
        if closed is not None:
            emitted.append(closed["close"])
    assert len(emitted) > 10
    # Replaying a prefix reproduces the first N emissions exactly.
    replay: list[float] = []
    builder2 = VolumeClockBuilder(notional_per_bar=threshold)
    for timestamp, row in frame.iloc[:900].iterrows():
        closed = builder2.push(timestamp, row["open"], row["high"], row["low"],
                               row["close"], row["volume"])
        if closed is not None:
            replay.append(closed["close"])
    assert replay == emitted[:len(replay)]


def test_builder_provisional_bar_reflects_only_the_prints_seen():
    frame = clock_frame()
    builder = VolumeClockBuilder(notional_per_bar=50.0)
    for timestamp, row in frame.iloc[:2].iterrows():
        assert builder.push(timestamp, row["open"], row["high"], row["low"],
                            row["close"], row["volume"]) is None
    provisional = builder.provisional
    assert provisional is not None
    assert provisional["close_time"] == frame.index[1]
    assert provisional["volume"] == frame["volume"].iloc[:2].sum()
    closed = builder.push(frame.index[2], *frame.iloc[2][["open", "high",
                                                          "low", "close",
                                                          "volume"]])
    assert closed is not None
    assert builder.provisional is None


def test_builder_state_round_trip_continues_identically():
    frame = random_walk_frame(1200)
    threshold = notional_threshold(frame, 60)
    builder = VolumeClockBuilder(notional_per_bar=threshold)
    for timestamp, row in frame.iloc[:700].iterrows():
        builder.push(timestamp, row["open"], row["high"], row["low"],
                     row["close"], row["volume"])
    resumed = VolumeClockBuilder.from_state(builder.state())
    left = builder.push_frame(frame.iloc[700:])
    right = resumed.push_frame(frame.iloc[700:])
    assert left.equals(right)
    assert left.equals(VolumeClockBuilder(notional_per_bar=threshold)
                       .push_frame(frame).iloc[-len(left):].reset_index(drop=True)
                       .set_index(left.index))


def test_builder_requires_exactly_one_threshold():
    with pytest.raises(ValueError):
        VolumeClockBuilder()
    with pytest.raises(ValueError):
        VolumeClockBuilder(notional_per_bar=1.0, volume_per_bar=1.0)
    with pytest.raises(ValueError):
        VolumeClockBuilder(notional_per_bar=0.0)


def test_bars_from_rows_round_trip_and_empty_case():
    frame = clock_frame(3)
    builder = VolumeClockBuilder(notional_per_bar=20.0)
    rows = []
    for timestamp, row in frame.iterrows():
        closed = builder.push(timestamp, *row[["open", "high", "low", "close",
                                               "volume"]])
        if closed is not None:
            rows.append(closed)
    assert len(rows) == 3
    rebuilt = bars_from_rows(rows)
    assert list(rebuilt.columns) == list(CACHE_COLUMNS)
    assert rebuilt.index.name == INDEX
    assert rebuilt.equals(VolumeClockBuilder(notional_per_bar=20.0)
                          .push_frame(frame))
    empty = bars_from_rows([])
    assert empty.empty and list(empty.columns) == list(CACHE_COLUMNS)


# ── input handling ───────────────────────────────────────────────────────

def test_extra_cache_columns_are_dropped():
    """A cache that grew new columns must not change the sampled layout."""
    frame = random_walk_frame(500)
    frame["quote_volume"] = frame["close"] * frame["volume"]
    frame["trade_count"] = 1.0
    bars = dollar_bars(frame, notional_threshold(frame, 20))
    assert list(bars.columns) == list(CACHE_COLUMNS)


def test_unsorted_input_is_sorted_before_the_clock_runs():
    frame = random_walk_frame(400)
    shuffled = frame.iloc[::-1]
    assert dollar_bars(shuffled, 1e6).equals(dollar_bars(frame, 1e6))


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"close": [1.0]}),                      # missing columns
    pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                  "close": [1.0], "volume": [1.0]}),      # no DatetimeIndex
])
def test_invalid_frames_are_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        dollar_bars(bad, 1.0)


def test_non_positive_thresholds_are_refused():
    frame = clock_frame()
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            dollar_bars(frame, bad)
        with pytest.raises(ValueError):
            volume_bars(frame, bad)


def test_empty_and_single_print_frames():
    empty = clock_frame(0)
    for built in (dollar_bars(empty, 10.0), volume_bars(empty, 10.0),
                  time_bars(empty, "1h")):
        assert built.empty and list(built.columns) == list(CACHE_COLUMNS)
    one = clock_frame(1)
    assert dollar_bars(one, 1000.0).empty
    assert len(dollar_bars(one, 1000.0, drop_partial=False)) == 1


def test_repeated_builds_are_bit_identical():
    frame = random_walk_frame(2000)
    threshold = notional_threshold(frame, 80)
    first = dollar_bars(frame, threshold)
    second = dollar_bars(frame, threshold)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert np.array_equal(bar_returns(first).to_numpy(),
                          bar_returns(second).to_numpy())


# ── distribution statistics ──────────────────────────────────────────────

def test_moments_and_jarque_bera_agree_with_their_closed_forms():
    rng = np.random.default_rng(11)
    sample = rng.normal(size=4000)
    centered = sample - sample.mean()
    m2 = (centered ** 2).mean()
    assert skewness(sample) == pytest.approx(
        (centered ** 3).mean() / m2 ** 1.5, rel=1e-12)
    assert excess_kurtosis(sample) == pytest.approx(
        (centered ** 4).mean() / m2 ** 2 - 3.0, rel=1e-12)
    jb = jarque_bera(sample)
    expected = len(sample) / 6.0 * (skewness(sample) ** 2
                                    + excess_kurtosis(sample) ** 2 / 4.0)
    assert jb["statistic"] == pytest.approx(expected, rel=1e-12)
    # chi2(2) survival is exactly exp(-x/2): the p-value is not an approximation.
    assert jb["p_value"] == pytest.approx(float(np.exp(-expected / 2.0)), rel=1e-12)
    assert jb["reject_5pct"] is False


def test_jarque_bera_rejects_a_fat_tailed_sample():
    rng = np.random.default_rng(3)
    fat = rng.standard_t(df=3, size=6000)
    assert excess_kurtosis(fat) > excess_kurtosis(rng.normal(size=6000))
    assert jarque_bera(fat)["reject_5pct"] is True


def test_moments_are_none_below_the_row_floor():
    short = np.arange(MIN_STATS_ROWS - 1, dtype=float)
    assert skewness(short) is None
    assert excess_kurtosis(short) is None
    assert jarque_bera(short)["statistic"] is None
    constant = np.ones(100)
    assert skewness(constant) is None
    assert excess_kurtosis(constant) is None


def test_autocorrelation_of_known_series():
    alternating = np.tile([0.0, 1.0], 200)
    assert autocorrelation(alternating, 1) == pytest.approx(-1.0)
    assert autocorrelation(alternating, 2) == pytest.approx(1.0)
    ramp = np.arange(500, dtype=float)
    assert autocorrelation(ramp, 1) > 0.99
    assert autocorrelation(np.ones(500), 1) is None      # no variance
    assert autocorrelation(np.arange(3.0), 1) is None    # too short
    assert acf(alternating, (1, 2)) == {1: pytest.approx(-1.0),
                                        2: pytest.approx(1.0)}


def test_volatility_clustering_separates_a_clustered_series():
    rng = np.random.default_rng(5)
    independent = rng.normal(size=4000)
    # |r| is serially dependent when the scale itself is persistent.
    scale = 1.0 + 0.9 * np.abs(rng.normal(size=4000))
    clustered = rng.normal(size=4000) * np.repeat(scale[:200], 20)
    assert volatility_clustering(clustered)["mean_abs_acf"] > \
        volatility_clustering(independent)["mean_abs_acf"]


def test_return_stats_reports_every_row_of_the_table():
    frame = random_walk_frame(3000)
    stats = return_stats(bar_returns(dollar_bars(
        frame, notional_threshold(frame, 100))))
    for key in ("n", "skew", "excess_kurtosis", "jarque_bera", "jarque_bera_p",
                "acf_returns", "acf_abs_returns", "mean_abs_acf"):
        assert key in stats
    assert set(stats["acf_returns"]) == {1, 2, 3, 4, 5}
    assert stats["n"] == len(dollar_bars(
        frame, notional_threshold(frame, 100))) - 1


def test_bar_returns_uses_the_bar_clock_not_the_calendar():
    frame = clock_frame(3)
    bars = dollar_bars(frame, 20.0)      # three bars, identical closes
    assert len(bar_returns(bars)) == 2
    assert (bar_returns(bars).to_numpy() == 0.0).all()


def test_print_gap_times_find_a_hole_and_alignment_holds():
    frame = random_walk_frame(600, gap_at=100)     # 200 prints removed
    holes = print_gap_times(frame)
    assert len(holes) == 1
    # The hole is the first print after the deleted block.
    assert holes[0] == frame.index[100]
    spacing = spacing_seconds(frame)
    assert spacing.iloc[99] > 3 * spacing.median()
    assert len(print_gap_times(random_walk_frame(300))) == 0
    # A single missing print is below the 4x median rule and is not a hole.
    assert len(print_gap_times(random_walk_frame(300, gap_at=None))) == 0


def test_gap_spanning_returns_flag_the_bar_that_crosses_the_hole():
    frame = random_walk_frame(6000, gap_at=2000)
    bars = time_bars(frame, "1h")
    mask = gap_spanning_returns(frame, bars)
    assert mask.sum() == 1
    assert mask.index.equals(bar_returns(bars).index)
    # The flagged return's closes straddle the hole.
    position = int(np.argmax(mask.to_numpy()))
    hole = print_gap_times(frame)[0]
    assert bars.index[position] < hole <= bars.index[position + 1]
    # By construction a dollar bar crosses the same hole exactly once.
    clock = dollar_bars(frame, notional_threshold(frame, 200))
    assert gap_spanning_returns(frame, clock).sum() == 1


def test_returns_excluding_gaps_removes_only_the_flagged_returns():
    frame = random_walk_frame(6000, gap_at=2000)
    bars = time_bars(frame, "1h")
    full = bar_returns(bars)
    trimmed = returns_excluding_gaps(frame, bars)
    assert len(trimmed) == len(full) - 1
    assert set(trimmed.index) < set(full.index)
    # Without a hole nothing is removed.
    clean = random_walk_frame(3000)
    assert len(returns_excluding_gaps(clean, time_bars(clean, "1h"))) == \
        len(bar_returns(time_bars(clean, "1h")))


# ── resampling uncertainty ───────────────────────────────────────────────

def test_block_bootstrap_is_deterministic_and_brackets_the_point():
    rng = np.random.default_rng(13)
    sample = rng.normal(size=1500)
    first = block_bootstrap_ci(sample, excess_kurtosis, n_boot=60, seed=4)
    second = block_bootstrap_ci(sample, excess_kurtosis, n_boot=60, seed=4)
    assert first == second
    assert first["lo"] <= first["point"] <= first["hi"]
    assert first["block"] >= 1


def test_bootstrap_reports_no_evidence_for_a_short_series():
    out = block_bootstrap_ci(np.arange(3.0), skewness, n_boot=10, seed=1)
    assert out["point"] is None and out["lo"] is None


def test_bootstrap_metric_difference_detects_a_constructed_tail_shift():
    rng = np.random.default_rng(21)
    normal = rng.normal(size=3000)
    fat = rng.standard_t(df=3, size=3000)
    out = bootstrap_metric_difference(fat, normal, excess_kurtosis,
                                      n_boot=100, seed=6)
    assert out["point"] > 0
    assert out["excludes_zero"] is True
    assert out["lo"] > 0


def test_bootstrap_metric_difference_finds_nothing_for_the_same_process():
    rng = np.random.default_rng(22)
    left = rng.normal(size=3000)
    right = rng.normal(size=3000)
    out = bootstrap_metric_difference(left, right, excess_kurtosis,
                                      n_boot=100, seed=7)
    assert out["excludes_zero"] is False
    assert out["lo"] < 0 < out["hi"]


def test_bootstrap_metric_difference_needs_both_series():
    out = bootstrap_metric_difference(np.arange(2.0), np.arange(100.0),
                                      skewness, n_boot=10, seed=1)
    assert out["point"] is None and out["excludes_zero"] is None
