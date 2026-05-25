import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    available_features = [f for f in feature_list if f in df.columns]
    if not available_features:
        return pd.DataFrame(index=df.index)

    feature_df = df[available_features].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().fillna(0)
    return feature_df


def create_label(df: pd.DataFrame, forward_periods: int = 4, threshold: float = 0.01) -> pd.Series:
    future_close = df["close"].shift(-forward_periods)
    return_pct = (future_close - df["close"]) / df["close"]
    labels = pd.Series("hold", index=df.index)
    labels[return_pct > threshold] = "up"
    labels[return_pct < -threshold] = "down"
    return labels


def create_binary_label(df: pd.DataFrame, forward_periods: int = 4) -> pd.Series:
    future_close = df["close"].shift(-forward_periods)
    labels = (future_close > df["close"]).astype(float)
    labels[labels.isna()] = np.nan
    return labels.astype("Int64")


def create_regression_label(df: pd.DataFrame, forward_periods: int = 4) -> pd.Series:
    future_close = df["close"].shift(-forward_periods)
    return_pct = (future_close - df["close"]) / df["close"]
    return return_pct
