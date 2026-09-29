"""代币检测 APIs — heuristic token screen over Binance public market data.

Contract: ``docs/overhaul/MARKET_PAGES_API.md`` §4.

    GET /api/audit/screen    ?limit=&min_quote_volume=&sort=&interval=
    GET /api/audit/{symbol}

Both are **reads**: any logged-in user (viewer included) may call them, matching
every other market endpoint in this app.  Nothing here writes to the database and
nothing here talks to a trading endpoint.

Honesty is part of the contract (``MARKET_PAGES_API.md`` §0/§4): the payload
carries a ``disclaimer`` string and the page renders it prominently, because
these numbers come from exchange market data only and are **not** an on-chain
contract audit.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.market_data.screener import (
    DISCLAIMER,
    UpstreamError,
    get_screener,
    normalize_min_quote_volume,
)

#: Hard ceiling for one request so a stalled host can never pin a worker.
ROUTE_TIMEOUT = 30.0

# --------------------------------------------------------------------------
# per-caller rate limiting (shared with web/routes/market.py)
# --------------------------------------------------------------------------
#: Requests allowed per window, per session (cookie) or client IP.  Both
#: `audit` and the fan-out market endpoints are reachable by any logged-in
#: user; without a ceiling an authenticated loop turns this process into a
#: crawler of the public exchange (one screen = ticker + depth + klines for up
#: to 200 symbols).  The budgets are deliberately generous — the UI polls a
#: couple of endpoints every few seconds — but a tight loop gets a 429.
RATE_LIMITS = {"audit": 30, "market": 300}
#: Sliding window for the limits above.
RATE_WINDOW_SEC = 60.0
#: Soft cap on tracked callers before stale buckets are pruned.
RATE_MAX_CALLERS = 512

#: (group, caller) -> list of request timestamps inside the window.
_rate_hits: dict[tuple[str, str], list[float]] = {}


def reset_rate_limits() -> None:
    """Drop all limiter state (tests; also called by the suite's fixtures)."""
    _rate_hits.clear()


def _rate_caller(request: Request) -> str:
    """Identify the caller: the session cookie when present, else the peer IP."""
    token = None
    try:
        token = request.cookies.get("bt_session")
    except Exception:
        token = None
    if token:
        return f"session:{token}"
    host = getattr(getattr(request, "client", None), "host", None)
    return f"ip:{host or 'unknown'}"


def _prune_rate_hits(now: float) -> None:
    if len(_rate_hits) <= RATE_MAX_CALLERS:
        return
    for key in [k for k, hits in _rate_hits.items()
                if not hits or now - hits[-1] >= RATE_WINDOW_SEC]:
        _rate_hits.pop(key, None)


def check_rate_limit(request: Request, group: str, limit: int = None,
                     window: float = None):
    """Record one hit; return a 429 ``JSONResponse`` once the budget is spent.

    Returns ``None`` when the caller is still inside its budget, so handlers can
    keep the ``if err := ...: return err`` shape used by the auth helpers.
    """
    limit = RATE_LIMITS.get(group, 60) if limit is None else limit
    window = RATE_WINDOW_SEC if window is None else window
    now = time.time()
    key = (group, _rate_caller(request))
    hits = [t for t in _rate_hits.get(key, []) if now - t < window]
    if len(hits) >= limit:
        _rate_hits[key] = hits
        return JSONResponse(
            {"error": f"Too many requests for '{group}', retry in a minute"},
            status_code=429)
    hits.append(now)
    _rate_hits[key] = hits
    _prune_rate_hits(now)
    return None


def register(app: FastAPI, ctx) -> None:
    config = ctx.config

    def _screener():
        return get_screener(config)

    def _error(message, status: int = 502):
        return JSONResponse({"error": str(message)}, status_code=status)

    def _method_note() -> dict:
        """Machine-readable statement of what the numbers mean (shown in the UI)."""
        return {
            "summary": "仅基于币安现货公开行情（24h ticker / depth / klines）的启发式筛查",
            "dimensions": [
                "流动性：24h 计价成交额、±1% 盘口深度、买卖价差",
                "波动性：30d 年化波动率、90d 最大回撤、单根 K 线异常涨跌次数",
                "价格质量：上下影线占比、连续同向 K 线、成交额活跃度代理",
                "上市时间：由最早一根日线推断",
                "集中度代理：平均单笔成交额（quote_volume / 成交笔数）",
            ],
            "not_covered": [
                "链上合约权限（mint / freeze / blacklist / proxy upgrade）",
                "持仓集中度与大户行为、资金池锁定、转移税",
                "蜜罐、貔貅盘、跑路等链上欺诈行为",
            ],
        }

    def _meta(extra: dict | None = None) -> dict:
        payload = {
            "disclaimer": DISCLAIMER,
            "heuristic": True,
            "on_chain_audit": False,
            "updated_at": int(time.time()),
            "host": getattr(_screener(), "host", None),
        }
        if extra:
            payload.update(extra)
        return payload

    @app.get("/api/audit/screen")
    async def audit_screen(request: Request, limit: int = 50, min_quote_volume: float = 100_000,
                           sort: str = "score", interval: str = "1h"):
        """Screen the top-``limit`` USDT pairs by 24h quote volume.

        ``limit`` is clamped to 1..200 (contract §4) and the whole screen is
        cached in-process for ~5 minutes, so a page reload is cheap while the
        first load pays for the upstream crawl.  ``min_quote_volume`` is
        normalised onto the screener's ladder before it reaches the cache key,
        so varying it cannot mint an unbounded number of crawls.
        """
        if err := check_rate_limit(request, "audit"):
            return err
        if sort not in ("score", "volume"):
            return JSONResponse(
                {"error": f"invalid sort '{sort}': expected score|volume"},
                status_code=400)
        interval = (interval or "1h").strip()
        if interval not in ("1h", "4h", "1d"):
            return JSONResponse(
                {"error": f"invalid interval '{interval}': expected 1h|4h|1d"},
                status_code=400)
        # Echo back the bucket that was actually used, not the raw float.
        min_quote_volume = normalize_min_quote_volume(min_quote_volume)

        try:
            payload = await asyncio.wait_for(
                _screener().screen(limit=limit, min_quote_volume=min_quote_volume,
                                   sort=sort, interval=interval),
                timeout=ROUTE_TIMEOUT)
        except asyncio.TimeoutError:
            return _error(
                f"检测超时（>{ROUTE_TIMEOUT:.0f}s）：行情主机响应过慢，请稍后重试", 504)
        except UpstreamError as exc:
            return _error(f"行情数据不可用: {exc}", 502)
        except Exception as exc:  # never leak a traceback
            return _error(f"检测失败: {type(exc).__name__}: {exc}", 500)

        results = payload.get("results", [])
        return _meta({
            "results": results,
            "count": payload.get("count", len(results)),
            "sampled": payload.get("sampled", len(results)),
            "universe_usdt": payload.get("universe_usdt"),
            "failures": payload.get("failures", []),
            "params": {
                "limit": limit,
                "min_quote_volume": min_quote_volume,
                "sort": sort,
                "interval": interval,
            },
            "cached": bool(payload.get("cached")),
            "method": _method_note(),
        })

    @app.get("/api/audit/{symbol}")
    async def audit_symbol(request: Request, symbol: str, interval: str = "1h"):
        """Full detail for one symbol: every metric plus each flag's evidence.

        Cached in the screener for ~5 minutes per ``(symbol, interval)``: the
        call costs five upstream requests, so repeating it in a loop used to be
        an uncached fan-out.
        """
        if err := check_rate_limit(request, "audit"):
            return err
        symbol = (symbol or "").strip().upper()
        if symbol == "SCREEN":
            # Defensive: /api/audit/screen is registered first, but this route
            # must never be able to interpret the literal path as a token.
            return JSONResponse({"error": "invalid symbol 'SCREEN'"}, status_code=400)
        if not symbol or not symbol.replace("-", "").isalnum() or len(symbol) > 24:
            return JSONResponse({"error": f"invalid symbol '{symbol}'"},
                                status_code=400)
        interval = (interval or "1h").strip()
        if interval not in ("1h", "4h", "1d"):
            return JSONResponse(
                {"error": f"invalid interval '{interval}': expected 1h|4h|1d"},
                status_code=400)

        try:
            detail = await asyncio.wait_for(_screener().detail(symbol, interval=interval),
                                            timeout=ROUTE_TIMEOUT)
        except asyncio.TimeoutError:
            return _error(
                f"检测超时（>{ROUTE_TIMEOUT:.0f}s）：行情主机响应过慢，请稍后重试", 504)
        except UpstreamError as exc:
            return _error(f"行情数据不可用: {exc}", 502)
        except Exception as exc:
            return _error(f"检测失败: {type(exc).__name__}: {exc}", 500)

        metrics = detail.get("metrics") or {}
        if not metrics.get("last") and not metrics.get("quote_volume"):
            return JSONResponse(
                {"error": f"未找到交易对 {symbol} 的行情数据（可能不是 USDT 现货交易对）"},
                status_code=404)
        return _meta({
            "symbol": detail["symbol"],
            "base_asset": detail.get("base_asset"),
            "quote_asset": detail.get("quote_asset"),
            "overall_score": detail.get("overall_score"),
            "risk_level": detail.get("risk_level"),
            "flag_messages": detail.get("flags", []),
            "flags": detail.get("flag_details", []),
            "metrics": metrics,
            "score_breakdown": detail.get("score_breakdown"),
            "method": _method_note(),
        })
