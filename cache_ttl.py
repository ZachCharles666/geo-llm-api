# cache_ttl.py
import time
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple


class TTLCache:
    """
    进程内存 TTL Cache（线程安全）
    - max_items: 防止无限增长
    - TTL: 秒
    - LRU: 超限时按最近最少使用淘汰
    """

    def __init__(self, max_items: int = 512):
        self.max_items = max(16, int(max_items))
        self._lock = threading.Lock()
        # key -> (expires_at, value)
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            exp, val = item
            if exp <= now:
                # expired
                try:
                    del self._data[key]
                except KeyError:
                    pass
                return None
            # refresh LRU
            self._data.move_to_end(key, last=True)
            return val

    def set(self, key: str, value: Any, ttl_sec: int) -> None:
        ttl_sec = int(ttl_sec)
        if ttl_sec <= 0:
            return
        exp = time.time() + ttl_sec
        with self._lock:
            self._data[key] = (exp, value)
            self._data.move_to_end(key, last=True)

            # prune expired first
            now = time.time()
            dead = [k for k, (e, _) in list(self._data.items()) if e <= now]
            for k in dead:
                try:
                    del self._data[k]
                except KeyError:
                    pass

            # enforce max_items (LRU)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            try:
                del self._data[key]
            except KeyError:
                pass

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._data)
