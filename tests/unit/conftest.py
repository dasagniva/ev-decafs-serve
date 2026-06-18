from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def welllog_fixture() -> tuple[np.ndarray, dict]:
    """Tiny (n=300) deterministic synthetic series mirroring the welllog generator."""
    y = pd.read_csv(FIXTURE_DIR / "welllog_fixture.csv")["welllog"].to_numpy(dtype=float)
    meta = json.loads((FIXTURE_DIR / "welllog_fixture_meta.json").read_text())
    return y, meta
