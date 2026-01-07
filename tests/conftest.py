import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure repository root is importable when pytest is invoked from other directories
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api_brand


@pytest.fixture
def app_instance():
    # Clear dependency overrides to ensure isolation between tests
    api_brand.app.dependency_overrides = {}
    return api_brand.app


@pytest.fixture
def client(app_instance):
    with TestClient(app_instance) as c:
        yield c
