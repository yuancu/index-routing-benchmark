#!/usr/bin/env python3
"""Extract small, reusable per-table profiles from Spider/BIRD SQLite files."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


CARDINALITY_CAP = 1000
TOP_VALUES_SIZE = 10
SAMPLE_CANDIDATES_PER_TABLE = 3
SAMPLE_STRING_MAX_CHARS = 150
TRUNCATED_STRING_SUFFIX = "...[truncated]"


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<binary:{len(value)} bytes>"
    if isinstance(value, str) and len(value) > SAMPLE_STRING_MAX_CHARS:
        return value[:SAMPLE_STRING_MAX_CHARS] + TRUNCATED_STRING_SUFFIX
    return value


def populated_count(row: Iterable[Any]) -> int:
    return sum(value is not None for value in row)


def user_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [str(row[0]) for row in rows]


def table_columns(
    connection: sqlite3.Connection, table: str
) -> list[tuple[str, str]]:
    rows = connection.execute(f"PRAGMA table_info({quote_identifier(table)})")
    return [(str(row[1]), str(row[2] or "").lower()) for row in rows]


def is_text_column(declared_type: str) -> bool:
    return not any(
        token in declared_type
        for token in (
            "int",
            "real",
            "float",
            "double",
            "decimal",
            "numeric",
            "bool",
            "date",
            "time",
            "blob",
        )
    )


def top_values(
    connection: sqlite3.Connection, table: str, column: str
) -> list[dict[str, Any]]:
    table_sql = quote_identifier(table)
    column_sql = quote_identifier(column)
    distinct = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT {column_sql}
            FROM {table_sql}
            WHERE {column_sql} IS NOT NULL
            LIMIT {CARDINALITY_CAP}
        )
        """
    ).fetchone()[0]
    if not 0 < distinct < CARDINALITY_CAP:
        return []
    rows = connection.execute(
        f"""
        SELECT {column_sql}, COUNT(*) AS value_count
        FROM {table_sql}
        WHERE {column_sql} IS NOT NULL
        GROUP BY {column_sql}
        ORDER BY value_count DESC, {column_sql}
        LIMIT {TOP_VALUES_SIZE}
        """
    )
    return [{"v": json_value(value), "c": int(count)} for value, count in rows]


def profile_table(connection: sqlite3.Connection, table: str) -> dict[str, Any]:
    table_sql = quote_identifier(table)
    columns = table_columns(connection, table)
    column_names = [column for column, _ in columns]
    richest: tuple[int, dict[str, Any]] | None = None  # (populated columns, row)
    if column_names:
        rows = connection.execute(
            f"SELECT * FROM {table_sql} LIMIT {SAMPLE_CANDIDATES_PER_TABLE}"
        )
        for row in rows:
            score = populated_count(row)
            if richest is None or score > richest[0]:
                richest = (
                    score,
                    {
                        column: json_value(value)
                        for column, value in zip(column_names, row)
                    },
                )

    field_top_values: dict[str, list[dict[str, Any]]] = {}
    for column, declared_type in columns:
        if is_text_column(declared_type):
            values = top_values(connection, table, column)
            if values:
                field_top_values[f"{column}.keyword"] = values

    return {
        "doc_count": int(
            connection.execute(f"SELECT COUNT(*) FROM {table_sql}").fetchone()[0]
        ),
        "samples": [richest[1]] if richest else [],
        "keyword_subfields": list(field_top_values),
        "field_top_values": field_top_values,
    }


def profile_database(path: Path) -> dict[str, Any]:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        return {
            table: profile_table(connection, table)
            for table in user_tables(connection)
        }
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = {}
    for position, path in enumerate(sorted(args.sqlite), start=1):
        snapshot[path.stem] = profile_database(path)
        print(f"[{position}/{len(args.sqlite)}] {path.stem}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
