#!/usr/bin/env python3
"""Build pairwise index-routing data.

Spider and BIRD: every table is one index; a table is positive when the
reference SQL references it. FeB4RAG: every resource is one index; labels come
from its released document judgments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import sqlglot
from sqlglot import exp


SPLITS = ("train", "eval", "test")

FEB4RAG_COLLECTIONS = [
    "arguana",
    "scidocs",
    "scifact",
    "dbpedia-entity",
    "signal1m",
    "trec-news",
    "fever",
    "climate-fever",
    "nfcorpus",
    "fiqa",
    "trec-covid",
    "nq",
    "hotpotqa",
    "robust04",
    "msmarco",
    "webis-touche2020",
]

# A FeB4RAG resource is positive when its best Top-10 document has this grade.
FEB4RAG_POSITIVE_GRADE = 2

NUMERIC_PROFILE_TYPES = {
    "long",
    "integer",
    "short",
    "byte",
    "double",
    "float",
    "half_float",
    "scaled_float",
    "unsigned_long",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError(f"Unsupported input format: {path}")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def shuffled(values: Iterable[str], seed: int, key: str) -> list[str]:
    values = list(values)
    random.Random(stable_seed(seed, key)).shuffle(values)
    return values


def relational_profile_type(column_type: Any) -> str:
    """Map SQL-oriented source types to the types used by the index profiler."""
    normalized = str(column_type or "").strip().lower()
    if any(token in normalized for token in ("date", "time", "timestamp")):
        return "date"
    if "bool" in normalized:
        return "boolean"
    if any(
        token in normalized
        for token in ("real", "float", "double", "decimal", "numeric")
    ):
        return "double"
    if any(token in normalized for token in ("int", "number")):
        return "long"
    return "text"


def retrieval_capabilities(mapping: dict[str, str]) -> dict[str, Any]:
    text_fields = []
    keyword_fields = []
    filter_fields = []
    date_fields = []
    vector_fields = []
    for field, field_type in mapping.items():
        if field_type == "text":
            text_fields.append(field)
        elif field_type == "keyword":
            keyword_fields.append(field)
            filter_fields.append(field)
        elif field_type == "date":
            date_fields.append(field)
            filter_fields.append(field)
        elif field_type == "boolean" or field_type in NUMERIC_PROFILE_TYPES:
            filter_fields.append(field)
        elif field_type == "knn_vector":
            vector_fields.append(field)
    return {
        "has_vector_field": bool(vector_fields),
        "vector_fields": vector_fields,
        "text_fields": text_fields,
        "keyword_fields": keyword_fields,
        "filter_fields": filter_fields,
        "date_fields": date_fields,
    }


def make_profile(
    index_name: str,
    mapping: dict[str, str],
    *,
    field_top_values: dict[str, list[dict[str, Any]]] | None = None,
    doc_count: int | None = None,
    samples: Sequence[dict[str, Any]] = (),
    provenance: dict[str, str],
) -> dict[str, Any]:
    """Create an OpenSearch-shaped profile and record each field's provenance."""
    return {
        "index": index_name,
        "mapping_compressed": mapping,
        "field_top_values": field_top_values or {},
        "time_ranges": {},
        "doc_count": doc_count,
        "samples": list(samples),
        "retrieval_capabilities": retrieval_capabilities(mapping),
        "default_pipelines": {},
        "profile_provenance": provenance,
    }


# --- Spider / BIRD: tables as indexes -------------------------------------


def make_table_profile(
    table_id: str, columns: list[dict[str, Any]], sqlite_table: dict[str, Any] | None
) -> dict[str, Any]:
    mapping = {
        column["name"]: relational_profile_type(column["type"]) for column in columns
    }
    if sqlite_table is None:
        return make_profile(
            table_id,
            mapping,
            provenance={
                "mapping_compressed": "source_table_schema",
                "field_top_values": "unavailable",
                "doc_count": "unavailable",
                "samples": "unavailable",
            },
        )
    for field in sqlite_table["keyword_subfields"]:
        mapping[field] = "keyword"
    return make_profile(
        table_id,
        mapping,
        field_top_values=sqlite_table["field_top_values"],
        doc_count=sqlite_table["doc_count"],
        samples=sqlite_table["samples"],
        provenance={
            "mapping_compressed": "source_table_schema_plus_sqlite_keyword_subfields",
            "field_top_values": "sqlite_low_cardinality_text_values",
            "doc_count": "sqlite_table_row_count",
            "samples": "sqlite_richest_of_first_three_rows",
        },
    )


def make_table_indexes(
    source: str,
    schema: dict[str, Any],
    routing_split: str,
    sqlite_tables: dict[str, dict[str, Any]],
    column_descriptions: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """Convert one official ``tables.json`` entry into table index records."""
    database_id = f"{source}:{schema['db_id']}"
    names = schema["table_names_original"]
    columns: list[list[dict[str, Any]]] = [[] for _ in names]
    column_owner = {}
    for position, ((table, name), (_, natural_name), column_type) in enumerate(
        zip(
            schema["column_names_original"],
            schema["column_names"],
            schema["column_types"],
        )
    ):
        if table < 0:
            continue
        column = {"name": name, "type": column_type}
        if natural_name.lower() != name.lower().replace("_", " "):
            column["natural_name"] = natural_name
        description = column_descriptions.get((names[table].lower(), name.lower()))
        if description:
            column["description"] = description
        columns[table].append(column)
        column_owner[position] = (table, name)

    primary_keys: dict[int, list[str]] = {}
    for key in schema["primary_keys"]:
        for position in key if isinstance(key, list) else [key]:
            table, name = column_owner[position]
            primary_keys.setdefault(table, []).append(name)
    foreign_keys: dict[int, list[dict[str, Any]]] = {}
    for position, referenced in schema["foreign_keys"]:
        (table, name), (ref_table, ref_name) = (
            column_owner[position],
            column_owner[referenced],
        )
        foreign_keys.setdefault(table, []).append(
            {
                "columns": [name],
                "referenced_index": f"{database_id}:{names[ref_table]}",
                "referenced_columns": [ref_name],
            }
        )

    sqlite_by_name = {name.lower(): profile for name, profile in sqlite_tables.items()}
    indexes = []
    for table, name in enumerate(names):
        # SQLite bookkeeping tables (e.g. sqlite_sequence) are not routable data.
        if name.lower().startswith("sqlite_"):
            continue
        if ":" in name:
            raise ValueError(f"{database_id}: table name {name!r} contains ':'")
        index_id = f"{database_id}:{name}"
        description = f"Table {name}"
        if schema["table_names"][table].lower() != name.lower().replace("_", " "):
            description += f" ({schema['table_names'][table]})"
        description += (
            f" with columns: {', '.join(c['name'] for c in columns[table])}."
        )
        indexes.append(
            {
                "index_id": index_id,
                "database_id": database_id,
                "source": source,
                "source_index_id": name,
                "routing_split": routing_split,
                "kind": "relational_table",
                "name": name,
                "description": description,
                "profile": make_table_profile(
                    index_id, columns[table], sqlite_by_name.get(name.lower())
                ),
                "schema": {
                    "columns": columns[table],
                    "primary_key": primary_keys.get(table, []),
                    "foreign_keys": foreign_keys.get(table, []),
                },
            }
        )
    if len({index["index_id"].lower() for index in indexes}) != len(indexes):
        raise ValueError(f"{database_id}: duplicate table names")
    return indexes


def referenced_tables(sql: str) -> set[str]:
    """Lower-cased physical table names referenced anywhere in the SQL."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    # ponytail: an unreferenced CTE's tables still count; scope analysis if CTEs appear.
    return {
        table.name.lower() for table in tree.find_all(exp.Table) if table.name
    } - cte_names


def build_sql_episode(
    query: dict[str, Any], table_ids: list[str], seed: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (episode, None) or (None, parse failure)."""
    failure = {
        "id": query["id"],
        "query": query["query"],
        "sql": query["sql"],
        "split": query["routing_split"],
    }
    try:
        used = referenced_tables(query["sql"])
    except sqlglot.errors.SqlglotError as exc:
        return None, {**failure, "reason": f"parse_error: {exc}"}
    by_name = {index_id.rsplit(":", 1)[1].lower(): index_id for index_id in table_ids}
    unknown = sorted(used - by_name.keys())
    if unknown:
        return None, {**failure, "reason": f"unknown_tables: {unknown}"}
    if not used:
        return None, {**failure, "reason": "no_physical_tables"}

    gold = {by_name[name] for name in used}
    candidates = shuffled(table_ids, seed, query["id"])
    return {
        "id": query["id"],
        "query": query["query"],
        "index": candidates,
        "label": [int(index_id in gold) for index_id in candidates],
        "sql": query["sql"],
        "database_id": query["database_id"],
    }, None


def assign_database_splits(
    source: str, databases_by_source_split: dict[str, set[str]], seed: int
) -> dict[str, str]:
    """Train databases stay train; held-out databases split evenly into eval/test."""
    splits = {db: "train" for db in databases_by_source_split["train"]}
    held_out = sorted(
        databases_by_source_split["dev"] - splits.keys(),
        key=lambda db: stable_seed(seed, f"routing-split:{source}:{db}"),
    )
    for position, db in enumerate(held_out):
        splits[db] = "eval" if position < len(held_out) // 2 else "test"
    return splits


def load_sql_queries(source: str, raw: Path) -> list[dict[str, Any]]:
    if source == "spider":
        inputs = [("train", "train.parquet"), ("dev", "validation.parquet")]
    else:
        inputs = [("train", "train_with_schema.json"), ("dev", "dev_with_schema.json")]
    queries = []
    for split, filename in inputs:
        for position, row in enumerate(read_rows(raw / filename)):
            query_number = row.get("question_id", position)
            queries.append(
                {
                    "id": f"{source}:{split}:{query_number}",
                    "query": row["question"],
                    "sql": row.get("query") or row.get("SQL"),
                    "database_id": f"{source}:{row['db_id']}",
                    "db_id": row["db_id"],
                    "source_split": split,
                }
            )
    return queries


def load_column_descriptions(raw: Path, db_id: str) -> dict[tuple[str, str], str]:
    """BIRD ``database_description/<table>.csv`` column descriptions."""
    descriptions = {}
    for path in sorted((raw / "database_description" / db_id).glob("*.csv")):
        data = path.read_bytes()
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:  # some BIRD CSVs are Windows-1252
            text = data.decode("cp1252", errors="replace")
        for row in csv.DictReader(text.splitlines()):
            row = {str(k).strip(): (v or "").strip() for k, v in row.items() if k}
            column = row.get("original_column_name", "")
            parts = [row.get("column_description", ""), row.get("value_description", "")]
            description = " ".join(part for part in parts if part)
            if column and description:
                descriptions[(path.stem.lower(), column.lower())] = description
    return descriptions


def build_sql_source(source: str, data_root: Path, seed: int) -> dict[str, Any]:
    raw = data_root / "raw" / source
    output_dir = data_root / source
    schemas = {
        schema["db_id"]: schema
        for path in sorted(raw.glob("*tables.json"))
        for schema in read_json(path)
    }
    sqlite_profiles = read_json(output_dir / "sqlite_profile_snapshot.json")
    queries = load_sql_queries(source, raw)

    databases_by_source_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    for query in queries:
        databases_by_source_split[query["source_split"]].add(query["db_id"])
    database_splits = assign_database_splits(source, databases_by_source_split, seed)

    indexes = []
    table_ids_by_database: dict[str, list[str]] = {}
    for db_id in sorted(database_splits):
        tables = make_table_indexes(
            source,
            schemas[db_id],
            database_splits[db_id],
            sqlite_profiles.get(db_id, {}),
            load_column_descriptions(raw, db_id),
        )
        indexes.extend(tables)
        table_ids_by_database[f"{source}:{db_id}"] = [t["index_id"] for t in tables]

    episodes, failures = [], []
    for query in queries:
        query["routing_split"] = database_splits[query["db_id"]]
        episode, failure = build_sql_episode(
            query, table_ids_by_database[query["database_id"]], seed
        )
        if episode:
            episode["_split"] = query["routing_split"]
            episodes.append(episode)
        else:
            failures.append(failure)
    write_jsonl(output_dir / "parse_failures.jsonl", failures)
    return write_source(
        source,
        output_dir,
        indexes,
        episodes,
        {
            "label_semantics": "reference_table_usage",
            "sql_parser": f"sqlglot=={sqlglot.__version__} (sqlite dialect)",
            "parse_failures": dict(Counter(f["split"] for f in failures)),
        },
    )


# --- FeB4RAG: resources as indexes ----------------------------------------


def read_feb4rag_document_grades(
    path: Path, query_count: int
) -> dict[tuple[str, str], int]:
    lines_per_collection = query_count * 10
    expected_lines = lines_per_collection * len(FEB4RAG_COLLECTIONS)
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if len(lines) != expected_lines:
        raise ValueError(
            f"{path} has {len(lines)} lines; expected {expected_lines}. "
            "The released file is collection-major with ten rows per query."
        )

    maximum_grades: dict[tuple[str, str], int] = {}
    offset = 0
    for resource in FEB4RAG_COLLECTIONS:
        for line in lines[offset : offset + lines_per_collection]:
            query_id, _, _, grade = line.split()
            key = (query_id, resource)
            maximum_grades[key] = max(maximum_grades.get(key, 0), int(grade))
        offset += lines_per_collection
    return maximum_grades


def make_feb4rag_index(row: dict[str, str], facts: dict[str, Any]) -> dict[str, Any]:
    resource = row["name"]
    doc_count = facts["doc_count"]
    vertical = row.get("vertical") or "unknown"
    task = row.get("Task") or "unknown"
    meta = {"resource": resource, "vertical": vertical, "task": task}
    mapping = {
        "_id": "keyword",
        "title": "text",
        "text": "text",
        "__meta.resource": "keyword",
        "__meta.vertical": "keyword",
        "__meta.task": "keyword",
    }
    for field in facts["keyword_subfields"]:
        mapping[field] = "keyword"
    top_values = {
        f"__meta.{key}": [{"v": value, "c": doc_count}] for key, value in meta.items()
    }
    top_values.update(facts["field_top_values"])
    return {
        "index_id": f"feb4rag:{resource}",
        "source": "feb4rag",
        "source_index_id": resource,
        "kind": "document_collection",
        "name": resource,
        "description": row.get("Description", ""),
        "profile": make_profile(
            resource,
            mapping,
            field_top_values=top_values,
            doc_count=doc_count,
            samples=[{**sample, "__meta": meta} for sample in facts["samples"]],
            provenance={
                "mapping_compressed": "beir_corpus_schema_plus_routing_metadata",
                "field_top_values": (
                    facts["field_top_values_source"] + "_plus_engine_metadata"
                ),
                "doc_count": facts["doc_count_source"],
                "samples": facts["sample_source"],
            },
        ),
        "vertical": row.get("vertical"),
        "task": row.get("Task"),
    }


def build_feb4rag(data_root: Path, seed: int) -> dict[str, Any]:
    dataset = data_root / "raw" / "feb4rag" / "dataset"
    output_dir = data_root / "feb4rag"
    facts = read_json(output_dir / "profile_snapshot.json")
    with (dataset / "engines" / "engines.csv").open(encoding="utf-8", newline="") as h:
        indexes = [make_feb4rag_index(row, facts[row["name"]]) for row in csv.DictReader(h)]

    ordered = sorted(
        indexes, key=lambda index: stable_seed(seed, f"routing-split:{index['index_id']}")
    )
    for position, index in enumerate(ordered):
        index["routing_split"] = "train" if position < 10 else "eval" if position < 13 else "test"
    split_by_index = {index["index_id"]: index["routing_split"] for index in indexes}

    origins = {}
    with (dataset / "queries" / "rid_mapping.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            query_id, resource, _ = line.rstrip("\n").split("\t")
            origins[query_id] = f"feb4rag:{resource}"
    requests = read_rows(dataset / "queries" / "requests.jsonl")
    grades = read_feb4rag_document_grades(
        dataset / "qrels" / "BEIR-QRELS-RM.txt", len(requests)
    )

    episodes = []
    for row in requests:
        query_id = str(row["_id"])
        split = split_by_index[origins[query_id]]
        episode_id = f"feb4rag:all:{query_id}"
        candidates = shuffled(
            (i for i in sorted(split_by_index) if split_by_index[i] == split),
            seed,
            episode_id,
        )
        max_grades = [grades[(query_id, i.split(":", 1)[1])] for i in candidates]
        episodes.append(
            {
                "id": episode_id,
                "query": row["text"],
                "index": candidates,
                "label": [int(g >= FEB4RAG_POSITIVE_GRADE) for g in max_grades],
                "max_document_grade": max_grades,
                "origin_index": origins[query_id],
                "_split": split,
            }
        )
    return write_source(
        "feb4rag",
        output_dir,
        indexes,
        episodes,
        {
            "label_semantics": (
                f"max_top10_document_grade >= {FEB4RAG_POSITIVE_GRADE}"
            )
        },
    )


# --- Output -----------------------------------------------------------------


def write_source(
    source: str,
    output_dir: Path,
    indexes: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    extra_summary: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    indexes = sorted(indexes, key=lambda index: index["index_id"])
    write_jsonl(output_dir / "indexes.jsonl", indexes)
    summary = {
        "source": source,
        "indexes": len(indexes),
        "indexes_with_rows": sum(bool(i["profile"]["doc_count"]) for i in indexes),
        **extra_summary,
        "splits": {},
    }
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for episode in episodes:
        by_split[episode.pop("_split")].append(episode)
    for split, rows in by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        labels = [label for row in rows for label in row["label"]]
        summary["splits"][split] = {
            "indexes": sum(i["routing_split"] == split for i in indexes),
            "episodes": len(rows),
            "pairs": len(labels),
            "positive_pairs": sum(labels),
            "all_positive_episodes": sum(all(row["label"]) for row in rows),
            "positives_per_episode": dict(
                sorted(Counter(sum(row["label"]) for row in rows).items())
            ),
        }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_data_root = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument(
        "--sources", nargs="+", default=["spider", "bird", "feb4rag"]
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {
        source: (
            build_feb4rag(args.data_root, args.seed)
            if source == "feb4rag"
            else build_sql_source(source, args.data_root, args.seed)
        )
        for source in args.sources
    }
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
