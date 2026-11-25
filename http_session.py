"""
http_session.py
---------------
HTTP sessions that can never wait forever.

requests has NO timeout by default: if a server accepts the connection
but never answers, the call blocks indefinitely. That is what froze the
full run twice on 25 Nov (stuck in the Open-Meteo live-forecast call,
whose client doesn't set a timeout). Every session used for Open-Meteo
now applies a default timeout to every request.
"""

import requests
import requests_cache

# (seconds to connect, seconds to wait for data). Ensemble calls with 15
# models can take ~5 s, so 60 s is generous; a stalled call fails loudly
# instead of hanging, and the retry wrapper tries again.
DEFAULT_TIMEOUT = (10, 60)


class _DefaultTimeout:
    def request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = DEFAULT_TIMEOUT
        return super().request(method, url, **kwargs)


class TimeoutSession(_DefaultTimeout, requests.Session):
    """requests.Session with a default timeout."""


class TimeoutCachedSession(_DefaultTimeout, requests_cache.CachedSession):
    """requests_cache.CachedSession with a default timeout."""
