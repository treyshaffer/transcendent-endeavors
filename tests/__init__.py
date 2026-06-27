"""Test suite. Forces the zero-setup in-memory backend so tests need no DB.

Run: python -m unittest discover -s tests -v
"""
import os

# Must be set before any `from src ...` import (config reads env at import time).
os.environ.setdefault("VECTOR_BACKEND", "memory")
os.environ.setdefault("SUBREDDIT", "japanweather")
