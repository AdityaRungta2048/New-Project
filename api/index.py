"""Vercel serverless entrypoint for the FastAPI service.

Vercel's @vercel/python runtime serves the ASGI ``app`` exported here. The whole
repo is available at build time, so we add the project root to sys.path and reuse
the exact same FastAPI app defined in ``arbiter/api.py`` — no duplication.

Storage is pointed at /tmp via vercel.json (the only writable path on Vercel);
the audit trail is therefore ephemeral in this environment, so retrieval/list/
analytics endpoints reflect only the current warm instance. For a persistent
deployment, host on a platform with a durable filesystem or point ARBITER_DB_PATH
at a mounted volume / external DB.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default storage to /tmp when running on Vercel (read-only project fs).
if os.environ.get("VERCEL") and not os.environ.get("ARBITER_DB_PATH"):
    os.environ["ARBITER_DB_PATH"] = "/tmp/arbitrations.sqlite"
    os.environ["ARBITER_JSON_DIR"] = "/tmp/arbitrations"

from arbiter.api import app  # noqa: E402  (import after sys.path setup)

# `app` is the ASGI application Vercel serves.
__all__ = ["app"]
