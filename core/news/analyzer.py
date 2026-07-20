import asyncio
import time
import numpy as np
import aiosqlite
from datetime import datetime
from typing import Optional
from loguru import logger

from app.event_bus import EventBus, Event, EventType
from app.config import Config
from core.market_data.provider import MarketDataProvider
from core.news.fetcher import NewsFetcher
from core.news.source_manager import NewsSourceManager
from db.database import get_db


class NewsAnalyzer:
    def __init__(self, config: Config, event_bus: EventBus, market_data: MarketDataProvider):
        self.config = config
        self.event_bus = event_bus
        self.market_data = market_data
        self.fetcher = NewsFetcher()
        self.source_manager = NewsSourceManager()
        self._deepseek_client = None
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._core_symbols: list[str] = []
        self._satellite_symbols: list[str] = []
        self._last_fetch_time: dict[str, float] = {}
        # Aggregate sentiment per symbol using EMA
        self._sentiment_ema: dict[str, float] = {}
        self._ema_alpha = 0.3  # weight for new sentiment value

    async def start(self, core_symbols: list[str] = None, satellite_symbols: list[str] = None):
        self._core_symbols = core_symbols or []
        self._satellite_symbols = satellite_symbols or []
        await self.fetcher.start()
        self._running = True
        self._tasks.append(asyncio.create_task(self._periodic_fetch()))
        self._tasks.append(asyncio.create_task(self._anomaly_monitor()))

    def set_symbols(self, core: list[str], satellite: list[str]):
        self._core_symbols = core
        self._satellite_symbols = satellite

    async def set_deepseek(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        try:
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"NewsAnalyzer: DeepSeek client configured for sentiment analysis")
        except ImportError:
            logger.warning("NewsAnalyzer: openai package not available")
        except Exception as e:
            logger.warning(f"NewsAnalyzer: DeepSeek init failed: {e}")

    async def _periodic_fetch(self):
        while self._running:
            try:
                all_symbols = self._core_symbols + self._satellite_symbols
                sources = await self.source_manager.get_all_enabled()

                if not sources:
                    # No external sources — use kline-based price momentum as sentiment proxy
                    await self._momentum_sentiment_fallback(all_symbols)
                    await asyncio.sleep(300)  # refresh every 5 min
                    continue

                for symbol in all_symbols:
                    for source in sources:
                        articles = await self.fetcher.fetch_from_source(
                            source, symbol, self.config.news_max_articles
                        )
                        for article in articles:
                            sentiment = await self._analyze_sentiment(article)
                            await self._save_article(
                                source.get("id", 0), symbol, article, sentiment
                            )

                interval = self.config.news_fetch_interval * 60
                await asyncio.sleep(interval)
            except Exception as e:
                logger.warning(f"News periodic fetch error: {e}")
                await asyncio.sleep(60)

    async def _momentum_sentiment_fallback(self, symbols: list[str]):
        """Use short-term price momentum from cached OHLCV data as sentiment proxy.
        Uses REST/WS kline cache — always available after pre-fetch."""
        produced = 0
        for sym in symbols:
            try:
                # Use 5m kline data (most responsive for scalp/day trading sentiment)
                df = await self.market_data.get_historical(sym, "5m", limit=20)
                if df is None or len(df) < 3:
                    continue

                closes = df["close"].values
                current = closes[-1]
                # Price changes over different lookback windows
                chg_5m = (current / closes[-2] - 1) * 100 if len(closes) >= 2 else None
                chg_15m = (current / closes[-4] - 1) * 100 if len(closes) >= 4 else None
                chg_1h = (current / closes[-13] - 1) * 100 if len(closes) >= 13 else None

                signals = [np.clip(chg_5m / 2.0, -1.0, 1.0) * 0.5]
                total_weight = 0.5
                if chg_15m is not None:
                    signals.append(np.clip(chg_15m / 4.0, -1.0, 1.0) * 0.3)
                    total_weight += 0.3
                if chg_1h is not None:
                    signals.append(np.clip(chg_1h / 8.0, -1.0, 1.0) * 0.2)
                    total_weight += 0.2

                sent = sum(signals) / total_weight if total_weight > 0 else 0
                sent = max(-1.0, min(1.0, sent))

                old_ema = self._sentiment_ema.get(sym, 0.0)
                new_ema = old_ema * (1 - self._ema_alpha) + sent * self._ema_alpha
                self._sentiment_ema[sym] = new_ema

                impact = self._classify_impact(abs(new_ema))
                await self.event_bus.publish(Event(EventType.NEWS_UPDATE, {
                    "symbol": sym,
                    "title": f"Price momentum: 5m={chg_5m:+.2f}% 15m={chg_15m:+.2f}% 1h={chg_1h:+.2f}%" if chg_5m else "Price momentum",
                    "sentiment": new_ema,
                    "impact_level": impact,
                }))
                produced += 1
                logger.info(f"Momentum sentiment {sym}: {new_ema:.3f} (5m={chg_5m:+.2f}% 15m={chg_15m:+.2f}% 1h={chg_1h:+.2f}%)")
            except Exception as e:
                logger.warning(f"Momentum sentiment failed for {sym}: {e}")
        if produced == 0:
            logger.info("Momentum sentiment: no kline data yet, will retry in 5 min")

    async def _anomaly_monitor(self):
        while self._running:
            try:
                for symbol in self._core_symbols + self._satellite_symbols:
                    price_change = self.market_data.get_price_change_pct(symbol, 5)
                    vol_ratio = await self.market_data.get_volume_ratio(symbol)

                    if price_change and abs(price_change) > self.config.anomaly_threshold_pct:
                        await self._trigger_emergency_fetch(symbol, f"Price change: {price_change:.2f}%")
                    elif vol_ratio and vol_ratio > self.config.volume_spike_multiplier:
                        await self._trigger_emergency_fetch(symbol, f"Volume spike: {vol_ratio:.1f}x")

                await asyncio.sleep(30)
            except Exception as e:
                logger.warning(f"News anomaly monitor error: {e}")
                await asyncio.sleep(5)

    async def _trigger_emergency_fetch(self, symbol: str, reason: str):
        now = time.time()
        if symbol in self._last_fetch_time and now - self._last_fetch_time[symbol] < 300:
            return  # 5 min cooldown per symbol to prevent alert storms
        self._last_fetch_time[symbol] = now

        sources = await self.source_manager.get_all_enabled()
        for source in sources:
            articles = await self.fetcher.fetch_from_source(source, symbol, 3)
            for article in articles:
                sentiment = await self._analyze_sentiment(article)
                await self._save_article(source.get("id", 0), symbol, article, sentiment)

        # Publish alert — consumed by AlertManager if subscribed
        await self.event_bus.publish(Event(EventType.NEWS_ALERT, {
            "symbol": symbol,
            "reason": reason,
            "message": f"Emergency fetch triggered for {symbol}: {reason}",
        }))

    async def _analyze_sentiment(self, article: dict) -> float:
        if self._deepseek_client:
            try:
                response = await self._deepseek_client.chat.completions.create(
                    model=self.config.ai_model,
                    messages=[{
                        "role": "system",
                        "content": "Analyze this news headline for cryptocurrency market sentiment. Return only a number between -1.0 (very bearish) and 1.0 (very bullish)."
                    }, {
                        "role": "user",
                        "content": f"Title: {article.get('title', '')}\nSummary: {article.get('content_summary', '')}"
                    }],
                    max_tokens=10,
                    temperature=0,
                )
                text = response.choices[0].message.content.strip()
                try:
                    return float(text)
                except ValueError:
                    return 0.0
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
        return 0.0

    async def _save_article(self, source_id: int, symbol: str,
                            article: dict, sentiment: float):
        db = await get_db()
        try:
            await db.execute(
                """INSERT INTO news_articles (source_id, symbol, title, url, content_summary, sentiment_score, published_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (source_id, symbol, article.get("title", ""), article.get("url", ""),
                 article.get("content_summary", ""), sentiment,
                 article.get("published_at", datetime.now().isoformat()))
            )
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to save news article: {e}")
        finally:
            await db.close()

        # Aggregate sentiment per symbol using EMA
        old_ema = self._sentiment_ema.get(symbol, 0.0)
        new_ema = old_ema * (1 - self._ema_alpha) + sentiment * self._ema_alpha
        self._sentiment_ema[symbol] = new_ema

        impact = self._classify_impact(abs(new_ema))
        await self.event_bus.publish(Event(EventType.NEWS_UPDATE, {
            "symbol": symbol,
            "title": article.get("title", ""),
            "sentiment": new_ema,
            "impact_level": impact,
        }))

    @staticmethod
    def _classify_impact(sentiment_abs: float) -> int:
        if sentiment_abs > 0.7:
            return 5
        elif sentiment_abs > 0.5:
            return 4
        elif sentiment_abs > 0.3:
            return 3
        elif sentiment_abs > 0.1:
            return 2
        return 1

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await self.fetcher.close()
