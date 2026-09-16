"""Validated, versioned table metadata shared by writers and readers."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ColumnSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    type: Literal["TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"]
    coercion_rule: dict[str, Any] | None = None
    raw_col: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    annotation: bool = False


class TableSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    columns: list[ColumnSchema]
    n_total_rows: int = Field(ge=0)
    n_data_rows: int = Field(ge=0)


def decode_table_schema(raw: str) -> TableSchema:
    """Read the only supported shape; legacy shapes require artifact migration."""
    return TableSchema.model_validate(json.loads(raw))
