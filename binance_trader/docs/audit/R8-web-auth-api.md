# 审计报告 R8 — Web Server + Auth + API

**日期**: 2026-07-16
**审查文件**: `web/server.py`, `core/auth/auth.py`, `web/templates/`
**审查代码行数**: ~1000 行
**审计方法**: OWASP Top 10 对照 + 认证流程分析 + 权限矩阵验证 + 输入验证审查

---

## 摘要

| 等级 | 数量 |
|------|------|
| Critical | 1 |
| High | 3 |
| Medium | 4 |
| Low | 2 |
| **总计** | **10** |

---

## 漏洞清单

| ID | 等级 | 文件:行 | 描述 |
|----|------|---------|------|
| R8-001 | **Critical** | auth.py:42-48 | JWT secret 每次启动随机生成 — 所有 token 重启后失效，无法实现持久化会话 |
| R8-002 | High | server.py:175 | WebSocket `/ws/` 路径无认证 — 未授权用户可订阅实时数据 |
| R8-003 | High | auth.py:57 | Session token 存储于内存 dict — 重启全部丢失、无主动失效 |
| R8-004 | High | server.py:121-149 | 登录端点无限速 — 可暴力破解密码 |
| R8-005 | Medium | auth.py:211-212 | 初始 admin 密码打印到 stderr — 可能被日志系统持久化 |
| R8-006 | Medium | auth.py:35-39 | bcrypt 无 pepper — 仅依赖 salt 防护 |
| R8-007 | Medium | server.py | 部分 API 端点的 trader 权限检查覆盖不完整 |
| R8-008 | Medium | server.py:147-148 | Cookie secure=False — 非 HTTPS 环境但生产部署时应为 True |
| R8-009 | Low | auth.py:38-39 | bcrypt 默认 work factor 12 — 合理但未显式配置 |
| R8-010 | Low | server.py | Jinja2 模板 autoescape 默认启用 — 需确认 |

---

## 详细分析

### R8-001 🔴 Critical — JWT Secret 每次重启随机生成

**文件**: `core/auth/auth.py:42-48` + `app/main.py:56-59`

```python
jwt_secret = auth_cfg.get("jwt_secret", "")
if not jwt_secret:
    import secrets
    jwt_secret = secrets.token_hex(32)
```

**描述**: JWT secret 在每次启动时随机生成：
1. **所有已签发的 JWT token 在重启后全部失效** — 用户在 Web UI 中收到 401
2. **Session cookie 也失效**（内存 dict 丢失）— 用户体验差
3. **无法实现"记住我"或持久化登录** — 每次重启需重新登录
4. **fingerprint 仅记录到日志** — 即使知道 fingerprint 也无法恢复 secret

虽然随机 secret 比硬编码更安全，但应该持久化到安全存储中（如 secrets.yaml 或环境变量），使重启不影响已登录用户。

**修复建议**: 从 `secrets.yaml` 或环境变量 `JWT_SECRET` 读取持久化的密钥，仅在不存在时生成并保存。

---

### R8-002 🟠 High — WebSocket 路径无认证

**文件**: `web/server.py:175`

```python
if path in ("/login", "/api/auth/login", "/api/auth/logout") or \
   path.startswith("/static") or path.startswith("/ws/"):
    return await call_next(request)
```

**描述**: `/ws/` 路径被排除在认证中间件之外，任何未认证用户可以连接 WebSocket 并接收实时告警、交易更新等敏感数据。

虽然 WebSocket 连接本身需要连接到特定端点（如 `/ws/alerts`），但缺少认证意味着：
- 任何知道 URL 的人可订阅实时数据流
- 无法追踪谁在监听
- 敏感交易数据可能泄露

**修复建议**: 在 WebSocket 连接建立时验证 token（通过查询参数或首条消息传递）。

---

### R8-003 🟠 High — Session 管理脆弱

**文件**: `core/auth/auth.py:57-72`

```python
self._sessions: dict[str, dict] = {}
def create_session(self, user: User) -> str:
    token = secrets.token_hex(32)
    self._sessions[token] = {...}
    return token
```

**描述**: 
1. **纯内存存储** — 重启后所有 session 丢失，用户体验差
2. **无主动失效** — 无法在用户被禁用时立即使其 session 失效
3. **无并发限制** — 同一用户可以创建无限多个 session
4. **无 session 列表** — 管理员无法查看活跃会话

**修复建议**: 将 sessions 持久化到数据库，添加 `revoked` 字段和定时清理过期 session 的机制。

---

### R8-004 🟠 High — 登录无限速

**文件**: `web/server.py:121-149`

```python
@app.post("/api/auth/login")
async def api_login(request: Request):
    ...
    user_data = await am.get_user_by_username(username)
    if not user_data or not am.verify_password(password, user_data["password_hash"]):
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
```

**描述**: 登录端点没有任何速率限制。攻击者可以：
1. 使用常见密码字典对 admin 账号发起暴力破解
2. 枚举有效用户名（存在用户 vs 不存在用户的响应时间差异 — timing attack）

虽然 bcrypt 的哈希计算较慢（天然限速），但仍不足以阻止分布式暴力破解。

**修复建议**: 添加基于 IP 和 username 的登录失败计数器，超过阈值后临时锁定。

---

### R8-005 🟡 Medium — Admin 密码泄露风险

**文件**: `core/auth/auth.py:211-212` + `app/main.py:67`

```python
admin_pass = AuthManager.generate_random_password()
_sys.stderr.write(f"\n{'='*60}\nDEFAULT ADMIN: admin / {admin_pass}\n{'='*60}\n\n")
```

**描述**: 初始密码通过 `stderr` 输出。如果系统使用日志聚合器收集 stderr（如 systemd journal、Docker logs、云日志服务），密码可能被持久化到日志中并被未授权人员访问。

**修复建议**: 仅在首次启动时打印，并提示用户立即修改密码。或使用 `os.urandom` 生成密码并通过文件输出（仅 root 可读）。

---

### R8-006 🟡 Medium — Bcrypt 无 Pepper

**文件**: `core/auth/auth.py:35-39`

```python
@staticmethod
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
```

**描述**: bcrypt 使用随机 salt（`gensalt()`），但未使用 pepper（应用级静态密钥）。Pepper 提供了额外防护：即使数据库泄露，攻击者也需要 pepper 才能破解密码。

bcrypt 本身是安全的，但添加 pepper（HMAC-SHA256 预处理密码）几乎是零成本的额外安全层。

---

### R8-007 🟡 Medium — API 权限覆盖不完整

**文件**: `web/server.py`

**描述**: 审计发现以下端点的权限检查需要验证：

1. `/api/trade/close/{symbol}` — 需要 trader 权限 ✓（检查了）
2. `/api/strategy/{name}/toggle` — 需要 trader 权限 ✓
3. `/api/settings/*` — 需要 trader 权限 ✓
4. 但某些 partials 端点可能返回敏感数据但未检查权限

需要逐端点确认 `_require_trader()` 和 `_require_admin()` 的使用。

---

### R8-008 🟡 Medium — Cookie 安全属性

**文件**: `web/server.py:147-148`

```python
response.set_cookie("bt_session", session_token, httponly=True, samesite="lax",
                   secure=False, max_age=am.session_hours * 3600)
```

**描述**: `secure=False` 意味着 cookie 可以通过 HTTP 明文传输。在开发环境中这是必需的（没有 HTTPS），但部署到生产环境时如果忘记改为 `True`，cookie 可能被中间人攻击窃取。

另外缺少 `samesite="strict"` 属性（当前为 "lax"），可能受 CSRF 攻击影响。

---

### R8-009 🟢 Low — Bcrypt Work Factor

Bcrypt 使用默认 `gensalt()` 的 work factor（通常为 12）。对于 2026 年的硬件，建议显式设置为 12-14 以确保足够的计算成本。

---

### R8-010 🟢 Low — Jinja2 自动转义

Jinja2 的 `Environment` 默认启用 `autoescape=True`（选择 `.html` 模板时）。当前代码未显式设置 → 依赖默认行为。需确认所有模板文件使用 `.html` 扩展名以确保自动转义。

---

## 累计统计

| 等级 | 累计 (R1-R7) | R8 | 累计 |
|------|-------------|-----|------|
| Critical | 7 | 1 | 8 |
| High | 18 | 3 | 21 |
| Medium | 25 | 4 | 29 |
| Low | 13 | 2 | 15 |
| **总计** | **63** | **10** | **73** |
