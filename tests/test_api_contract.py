"""Stable API contract version for clients."""

import re

from app.api_contract import API_SCHEMA_VERSION


def test_api_schema_version_semver():
    assert re.match(r"^\d+\.\d+\.\d+$", API_SCHEMA_VERSION)
