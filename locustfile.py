"""Load test for the EV-DeCAFS serving API (one short run; numbers recorded in the README).

Run against a running API (container or local uvicorn):

    uvx locust -f locustfile.py --headless -u 20 -r 5 -t 30s --host http://localhost:8000

Posts the example well-log payload to /detect and occasionally hits /health and
/monitoring/drift, mirroring realistic traffic.
"""

from __future__ import annotations

import json
from pathlib import Path

from locust import HttpUser, between, task

_PAYLOAD = json.loads((Path(__file__).parent / "examples" / "welllog.json").read_text())


class EvDecafsUser(HttpUser):
    wait_time = between(0.0, 0.1)

    @task(10)
    def detect(self) -> None:
        self.client.post("/detect", json=_PAYLOAD)

    @task(1)
    def health(self) -> None:
        self.client.get("/health")

    @task(1)
    def drift(self) -> None:
        self.client.get("/monitoring/drift")
