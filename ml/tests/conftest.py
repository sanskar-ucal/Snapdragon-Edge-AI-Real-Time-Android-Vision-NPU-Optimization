import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Variant  # noqa: E402


def require(*variants: Variant):
    missing = [v.name for v in variants if not v.path.exists()]
    if missing:
        pytest.skip(f"models not built (run scripts/make_models.sh): {missing}")
