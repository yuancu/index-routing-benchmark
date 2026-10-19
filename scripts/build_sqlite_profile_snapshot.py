#!/usr/bin/env python3
"""Extract small, reusable profile overrides from Spider/BIRD SQLite ZIPs."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
import zipfile
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


def profile_database(path: Path) -> dict[str, Any]:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = user_tables(connection)
        document_count = 0
        richest: tuple[int, dict[str, Any]] | None = None
        field_top_values: dict[str, list[dict[str, Any]]] = {}
        keyword_subfields = []

        for table in tables:
            table_sql = quote_identifier(table)
            document_count += int(
                connection.execute(f"SELECT COUNT(*) FROM {table_sql}").fetchone()[0]
            )
            columns = table_columns(connection, table)
            column_names = [column for column, _ in columns]
            if column_names:
                rows = connection.execute(
                    f"SELECT * FROM {table_sql} LIMIT {SAMPLE_CANDIDATES_PER_TABLE}"
                )
                for row in rows:
                    candidate = {
                        "_table": table,
                        table: {
                            column: json_value(value)
                            for column, value in zip(column_names, row)
                        },
                    }
                    score = populated_count(row)
                    if richest is None or score > richest[0]:
                        richest = (score, candidate)

            for column, declared_type in columns:
                if not is_text_column(declared_type):
                    continue
                values = top_values(connection, table, column)
                if values:
                    field = f"{table}.{column}.keyword"
                    keyword_subfields.append(field)
                    field_top_values[field] = values

        return {
            "doc_count": document_count,
            "samples": [richest[1]] if richest else [],
            "keyword_subfields": keyword_subfields,
            "field_top_values": field_top_values,
        }
    finally:
        connection.close()


def sqlite_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return sorted(
        (
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".sqlite")
        ),
        key=lambda member: member.filename,
    )


def build_snapshot(archive_path: Path) -> dict[str, dict[str, Any]]:
    snapshot = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = sqlite_members(archive)
        for position, member in enumerate(members, start=1):
            index_name = Path(member.filename).stem
            with tempfile.TemporaryDirectory(prefix="index-profile-") as temp_dir:
                sqlite_path = Path(temp_dir) / f"{index_name}.sqlite"
                with archive.open(member) as source, sqlite_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                snapshot[index_name] = profile_database(sqlite_path)
            print(f"[{position}/{len(members)}] {index_name}")
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive", type=Path)
    inputs.add_argument("--sqlite", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge new profiles into an existing output snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = {}
    if args.merge and args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
    if args.archive:
        snapshot.update(build_snapshot(args.archive))
    else:
        for path in args.sqlite:
            snapshot[path.stem] = profile_database(path)
            print(path.stem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
