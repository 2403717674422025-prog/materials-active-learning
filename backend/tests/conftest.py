"""Pytest configuration and shared fixtures."""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (skipped by default)")
