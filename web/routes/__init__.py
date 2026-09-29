"""Package for the split web route modules.

Each module exposes ``register(app, ctx)`` and attaches its handlers to the
FastAPI app passed in by ``web.server.create_app()``.
"""
