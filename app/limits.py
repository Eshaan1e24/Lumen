"""Operational limits for a small, free-tier deployment: session store, rate limiting, upload checks.

Everything is in-process and in-memory (single worker), matching the privacy promise that files are never
written to disk. Tunable through environment variables so a self-hoster with more RAM can raise them.
"""
import os, io, time, threading, zipfile
from collections import OrderedDict, deque


def _env(name: str, default: float) -> float:
    try: return float(os.getenv(name, default))
    except ValueError: return float(default)


MAX_UPLOAD_BYTES = int(_env("LUMEN_MAX_UPLOAD_MB", 5) * 1024 * 1024)   # Render's free tier has 512 MB RAM in total
MAX_ROWS = int(_env("LUMEN_MAX_ROWS", 200_000))
MAX_COLS = int(_env("LUMEN_MAX_COLS", 200))
MAX_UNZIPPED_BYTES = int(_env("LUMEN_MAX_UNZIPPED_MB", 60) * 1024 * 1024)  # .xlsx is a zip: stop decompression bombs
MAX_SESSIONS = int(_env("LUMEN_MAX_SESSIONS", 40))
MAX_CELLS = int(_env("LUMEN_MAX_CELLS", 3_000_000))     # rows x columns held across all sessions: the real memory limit
SESSION_TTL = _env("LUMEN_SESSION_TTL_S", 1800)


class UploadTooLarge(ValueError): pass
class UploadRejected(ValueError): pass


async def read_capped(file, limit: int | None = None, chunk: int = 1 << 20) -> bytes:
    """Read an UploadFile but stop as soon as it exceeds `limit` (the old code buffered the whole body first)."""
    limit = MAX_UPLOAD_BYTES if limit is None else limit
    buf, size = bytearray(), 0
    while True:
        part = await file.read(chunk)
        if not part: break
        size += len(part)
        if size > limit: raise UploadTooLarge(f"File is larger than {limit // (1024 * 1024)} MB.")
        buf += part
    return bytes(buf)


def check_xlsx_bomb(raw: bytes) -> None:
    """Reject a .xlsx whose decompressed size is absurd compared with the upload (zip bomb)."""
    if raw[:4] != b"PK\x03\x04": return
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            total = sum(i.file_size for i in z.infolist())
    except zipfile.BadZipFile: return   # not a zip after all; the normal reader will give the real error
    if total > MAX_UNZIPPED_BYTES:
        raise UploadRejected(f"That spreadsheet expands to {total // (1024 * 1024)} MB, which is more than this server accepts.")


class SessionStore:
    """LRU + TTL store, bounded by count AND by total size (rows x columns), so many small sessions fit but a few huge uploads cannot
    exhaust memory. Using a session refreshes it, so an active user is never evicted by newcomers."""
    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl: float = SESSION_TTL, clock=time.monotonic, max_weight: int | None = None):
        self.max, self.ttl, self.clock, self.max_weight = max_sessions, ttl, clock, max_weight
        self._d: OrderedDict = OrderedDict()   # sid -> (last_used, value, weight)
        self._lock = threading.Lock()

    def _purge(self, now: float):
        for sid in [s for s, (t, _, _) in self._d.items() if now - t > self.ttl]: del self._d[sid]

    def put(self, sid: str, value, weight: int = 1) -> None:
        with self._lock:
            now = self.clock(); self._purge(now)
            self._d[sid] = (now, value, weight); self._d.move_to_end(sid)
            while len(self._d) > self.max or (self.max_weight and len(self._d) > 1 and sum(w for _, _, w in self._d.values()) > self.max_weight):
                self._d.popitem(last=False)

    def get(self, sid: str):
        with self._lock:
            now = self.clock(); self._purge(now)
            if sid not in self._d: raise KeyError(sid)
            _, value, weight = self._d[sid]; self._d[sid] = (now, value, weight); self._d.move_to_end(sid)
            return value

    def __len__(self): return len(self._d)


class RateLimiter:
    """Sliding-window limiter per key, plus a global daily budget so a burst of judges cannot drain the
    owner's Gemini quota. `check` returns (allowed, retry_after_seconds, reason)."""
    def __init__(self, per_window: int = 8, window: float = 600, per_day: int = 60, global_per_day: int = 400, clock=time.monotonic):
        self.per_window, self.window, self.per_day, self.global_per_day, self.clock = per_window, window, per_day, global_per_day, clock
        self._hits: dict[str, deque] = {}
        self._day_start, self._global = clock(), 0
        self._lock = threading.Lock()

    def check(self, key: str):
        with self._lock:
            now = self.clock()
            if now - self._day_start > 86400:
                self._day_start, self._global = now, 0
                self._hits = {k: q for k, q in self._hits.items() if q and now - q[-1] < 86400}
            if self._global >= self.global_per_day: return False, int(86400 - (now - self._day_start)), "global"
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > 86400: q.popleft()
            if len(q) >= self.per_day: return False, int(86400 - (now - q[0])), "daily"
            recent = [t for t in q if now - t <= self.window]
            if len(recent) >= self.per_window: return False, int(self.window - (now - recent[0])) + 1, "window"
            q.append(now); self._global += 1
            return True, 0, ""


def client_key(request) -> str:
    """Best-effort client id. Behind a proxy (Render) the socket peer is the proxy, so use X-Forwarded-For when
    LUMEN_TRUST_PROXY=1. A spoofed header can dodge the per-client limit but not the global daily budget."""
    if os.getenv("LUMEN_TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd: return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
