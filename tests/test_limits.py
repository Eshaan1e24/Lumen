import asyncio, io, zipfile
import pytest
from app import limits


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def test_session_store_is_lru_not_fifo():
    c = Clock(); s = limits.SessionStore(max_sessions=2, ttl=1000, clock=c)
    s.put("a", 1); s.put("b", 2)
    assert s.get("a") == 1          # touch a: now b is the least recently used
    s.put("c", 3)
    assert s.get("a") == 1 and s.get("c") == 3
    with pytest.raises(KeyError): s.get("b")


def test_session_store_expires():
    c = Clock(); s = limits.SessionStore(max_sessions=5, ttl=60, clock=c)
    s.put("a", 1); c.t += 61
    with pytest.raises(KeyError): s.get("a")


def test_active_session_survives_many_newcomers():
    c = Clock(); s = limits.SessionStore(max_sessions=3, ttl=1000, clock=c)
    s.put("mine", "x")
    for i in range(50):
        s.put(f"other{i}", i); c.t += 1
        assert s.get("mine") == "x"   # in use, so never evicted


def test_rate_limiter_window_daily_and_global():
    c = Clock(); r = limits.RateLimiter(per_window=3, window=600, per_day=5, global_per_day=7, clock=c)
    assert all(r.check("ip1")[0] for _ in range(3))
    ok, retry, why = r.check("ip1"); assert not ok and why == "window" and 0 < retry <= 601
    c.t += 601
    assert r.check("ip1")[0] and r.check("ip1")[0]          # 5th hit of the day
    c.t += 601
    ok, _, why = r.check("ip1"); assert not ok and why == "daily"
    assert r.check("ip2")[0]                                 # 6th global hit
    assert r.check("ip3")[0]                                 # 7th = the last one in the budget
    ok, _, why = r.check("ip4"); assert not ok and why == "global"
    c.t += 86401
    assert r.check("ip4")[0]                                 # new day


class FakeUpload:
    def __init__(self, data): self._b = io.BytesIO(data)
    async def read(self, n=-1): return self._b.read(n)


def test_read_capped_stops_early():
    big = FakeUpload(b"x" * (3 * 1024 * 1024))
    with pytest.raises(limits.UploadTooLarge): asyncio.run(limits.read_capped(big, limit=1024 * 1024))
    assert asyncio.run(limits.read_capped(FakeUpload(b"hello"), limit=10)) == b"hello"


def test_zip_bomb_rejected(monkeypatch):
    monkeypatch.setattr(limits, "MAX_UNZIPPED_BYTES", 1024 * 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z: z.writestr("sheet.xml", b"0" * (5 * 1024 * 1024))
    assert len(buf.getvalue()) < 50_000                     # tiny on the wire
    with pytest.raises(limits.UploadRejected): limits.check_xlsx_bomb(buf.getvalue())
    limits.check_xlsx_bomb(b"a,b\n1,2\n")                   # CSV passes untouched


def test_session_store_is_bounded_by_size_not_just_count():
    c = Clock(); s = limits.SessionStore(max_sessions=100, ttl=1000, clock=c, max_weight=1000)
    for i in range(20): s.put(f"small{i}", i, weight=10); c.t += 1             # twenty small sessions fit
    assert len(s) == 20 and s.get("small0") == 0
    s.put("huge", "x", weight=900); c.t += 1                                   # a huge one evicts the oldest, never the newest
    assert s.get("huge") == "x" and len(s) < 20
