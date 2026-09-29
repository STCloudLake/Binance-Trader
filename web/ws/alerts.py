"""``/ws/alerts`` real-time alert push WebSocket."""
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


def register(app: FastAPI, ctx) -> None:
    # ---- WebSocket for real-time alert push ----
    @app.websocket("/ws/alerts")
    async def ws_alerts(websocket: WebSocket):
        # Verify authentication via query parameter token
        token = websocket.query_params.get("token")
        am = getattr(app.state, "auth_manager", None)
        if am and token:
            session = am.verify_session(token)
            if not session:
                payload = am.verify_jwt(token)
                if not payload:
                    await websocket.close(code=4001, reason="Unauthorized")
                    return
        elif am:
            await websocket.close(code=4001, reason="Authentication required")
            return

        await websocket.accept()
        mgr = getattr(app.state, "alert_manager", None)
        if not mgr:
            await websocket.close()
            return
        client_id = str(id(websocket))
        queue = mgr.register_ws(client_id)
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    await websocket.send_text(msg)
                except asyncio.TimeoutError:
                    await websocket.send_text('{"ping": true}')
        except WebSocketDisconnect:
            pass
        finally:
            mgr.unregister_ws(client_id)
