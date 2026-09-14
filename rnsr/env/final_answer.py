"""FINAL short-circuit exception, shared by the child and docdb tools.

Imported from a stable module so `except FinalAnswer` is an identity
check even when sandbox_child runs as ``__main__``.
"""

from __future__ import annotations


class FinalAnswer(Exception):
    def __init__(self, value, is_var: bool, verification: dict | None = None):
        self.value = value
        self.is_var = is_var
        self.verification = verification
