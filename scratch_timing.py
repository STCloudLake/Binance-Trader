import time, pandas as pd
from core.ml.volatility import forecast_vol
df = pd.read_parquet("data/market/BTCUSDT/1h.parquet").tail(600)
forecast_vol(df)
t0=time.perf_counter()
for _ in range(20): forecast_vol(df)
print(f"forecast_vol(600 bars, window=500): {(time.perf_counter()-t0)/20*1e3:.3f} ms")
t0=time.perf_counter()
for _ in range(20): forecast_vol(df, window=0)
print(f"forecast_vol(600 bars, window=0):   {(time.perf_counter()-t0)/20*1e3:.3f} ms")
