#!/usr/bin/env python3
"""Build source-grouped index-routing data from Spider, BIRD, and FeB4RAG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import bm25s


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
            value = json.load(handle)
        if isinstance(value, list):
            return value
        raise ValueError(f"{path} must contain a JSON array")
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow; install requirements.txt"
            ) from exc
        return pq.read_table(path).to_pylist()
    raise ValueError(f"Unsupported input format: {path}")


def iter_input_rows(paths: Sequence[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in paths:
        split = infer_split(path)
        for row in read_rows(path):
            yield split, row


def infer_split(path: Path) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "validation" in name:
        return "dev"
    if "test" in name:
        return "test"
    return "unknown"


def normalize_text(value: str) -> str:
    value = value.replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def make_description(name: str, tables: Sequence[dict[str, Any]]) -> str:
    table_names = [str(table.get("name", "")).strip() for table in tables]
    table_names = [name for name in table_names if name]
    if table_names:
        return f"Database with tables: {', '.join(table_names)}."
    return f"Dataset about {normalize_text(name)}."


def make_retrieval_text(
    name: str, description: str, tables: Sequence[dict[str, Any]]
) -> str:
    parts = [normalize_text(name), description]
    for table in tables:
        parts.append(normalize_text(str(table.get("name", ""))))
        for column in table.get("columns", []):
            parts.append(normalize_text(str(column.get("name", ""))))
            column_description = column.get("description")
            if column_description:
                parts.append(str(column_description))
    return normalize_text(" ".join(parts))


def parse_official_schema(row: dict[str, Any]) -> list[dict[str, Any]]:
    table_names = row.get("table_names_original") or row.get("table_names") or []
    column_names = row.get("column_names_original") or row.get("column_names") or []
    column_types = row.get("column_types") or []
    tables = [{"name": name, "columns": []} for name in table_names]
    for position, column in enumerate(column_names):
        if not isinstance(column, (list, tuple)) or len(column) != 2:
            continue
        table_position, column_name = column
        if table_position is None or int(table_position) < 0:
            continue
        table_position = int(table_position)
        if table_position >= len(tables):
            continue
        column_type = column_types[position] if position < len(column_types) else None
        tables[table_position]["columns"].append(
            {"name": column_name, "type": column_type}
        )
    return tables


def parse_string_schema(schema_text: str) -> list[dict[str, Any]]:
    tables = []
    for table_part in schema_text.split("|"):
        if ":" not in table_part:
            continue
        table_name, columns_text = table_part.split(":", 1)
        columns = []
        for column_text in columns_text.split(","):
            column_text = column_text.strip()
            match = re.match(r"(.+?)\s*\(([^()]*)\)\s*$", column_text)
            if match:
                columns.append(
                    {"name": match.group(1).strip(), "type": match.group(2).strip()}
                )
            elif column_text:
                columns.append({"name": column_text, "type": None})
        tables.append({"name": table_name.strip(), "columns": columns})
    return tables


def schema_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    schema = row.get("schema")
    if isinstance(schema, dict):
        return parse_official_schema(schema)
    if "table_names_original" in row or "table_names" in row:
        return parse_official_schema(row)
    string_schema = row.get("Schema (values (type))")
    if isinstance(string_schema, str):
        return parse_string_schema(string_schema)
    return []


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
    provenance: dict[str, str] | None = None,
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
        "profile_provenance": provenance or {
            "mapping_compressed": "source_schema",
            "field_top_values": "unavailable",
            "doc_count": "unavailable",
            "samples": "unavailable",
        },
    }


def make_relational_profile(
    source_index_id: str,
    tables: Sequence[dict[str, Any]],
    sqlite_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mapping = {
        "__meta.table": "keyword",
        "__meta.column_type": "keyword",
    }
    type_table_counts: defaultdict[str, int] = defaultdict(int)
    for table in tables:
        table_name = str(table.get("name", "")).strip()
        table_types = set()
        for column in table.get("columns", []):
            column_name = str(column.get("name", "")).strip()
            if table_name and column_name:
                mapping[f"{table_name}.{column_name}"] = relational_profile_type(
                    column.get("type")
                )
            source_type = str(column.get("type") or "unknown").strip().lower()
            table_types.add(source_type)
        for source_type in table_types:
            type_table_counts[source_type] += 1

    field_top_values: dict[str, list[dict[str, Any]]] = {
        "__meta.table": [
            {"v": str(table.get("name", "")), "c": 1}
            for table in tables[:10]
            if table.get("name")
        ],
        "__meta.column_type": [
            {"v": source_type, "c": count}
            for source_type, count in sorted(
                type_table_counts.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        ],
    }
    richest_table = max(
        tables, key=lambda table: len(table.get("columns", [])), default=None
    )
    samples = []
    if richest_table:
        samples.append(
            {
                "_sample_type": "table_schema",
                "__meta": {
                    "table": richest_table.get("name"),
                    "column_type": sorted(
                        {
                            str(column.get("type") or "unknown").strip().lower()
                            for column in richest_table.get("columns", [])
                        }
                    ),
                },
                "table": richest_table.get("name"),
                "fields": {
                    str(column.get("name", "")): column.get("type")
                    for column in richest_table.get("columns", [])
                    if column.get("name")
                },
            }
        )
    doc_count = len(tables)
    provenance = {
        "mapping_compressed": "source_database_schema_plus_routing_metadata",
        "field_top_values": "source_schema_low_cardinality_metadata",
        "doc_count": "source_schema_table_count",
        "samples": "source_schema_richest_table",
    }
    if sqlite_profile and sqlite_profile["doc_count"] > 0:
        for field in sqlite_profile["keyword_subfields"]:
            mapping[field] = "keyword"
        field_top_values.update(sqlite_profile["field_top_values"])
        doc_count = sqlite_profile["doc_count"]
        samples = sqlite_profile["samples"]
        provenance.update(
            {
                "mapping_compressed": (
                    "source_database_schema_plus_sqlite_keyword_subfields"
                ),
                "field_top_values": "sqlite_low_cardinality_text_values",
                "doc_count": "sqlite_total_rows_across_tables",
                "samples": "sqlite_richest_of_first_three_rows_per_table",
            }
        )

    return make_profile(
        source_index_id,
        mapping,
        field_top_values=field_top_values,
        doc_count=doc_count,
        samples=samples,
        provenance=provenance,
    )


def make_relational_index(
    source: str,
    source_index_id: str,
    schema_row: dict[str, Any],
    sqlite_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tables = schema_from_row(schema_row)
    name = normalize_text(source_index_id)
    description = make_description(name, tables)
    return {
        "index_id": f"{source}:{source_index_id}",
        "source": source,
        "source_index_id": source_index_id,
        "kind": "relational_database",
        "name": name,
        "description": description,
        "retrieval_text": make_retrieval_text(name, description, tables),
        "profile": make_relational_profile(
            source_index_id, tables, sqlite_profiles.get(source_index_id)
        ),
        "schema": {
            "tables": tables,
            "primary_keys": schema_row.get("primary_keys")
            or schema_row.get("Primary Keys"),
            "foreign_keys": schema_row.get("foreign_keys")
            or schema_row.get("Foreign Keys"),
        },
    }


def load_schema_rows(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_rows(path):
            source_index_id = row.get("db_id")
            if source_index_id:
                schemas[str(source_index_id)] = row
    return schemas


def load_spider(
    question_paths: Sequence[Path],
    schema_paths: Sequence[Path],
    sqlite_profile_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schemas = load_schema_rows(schema_paths)
    with sqlite_profile_path.open(encoding="utf-8") as handle:
        sqlite_profiles = json.load(handle)
    indexes: dict[str, dict[str, Any]] = {}
    queries = []
    counters: defaultdict[str, int] = defaultdict(int)

    for split, row in iter_input_rows(question_paths):
        db_id = str(row["db_id"])
        index_id = f"spider:{db_id}"
        schema_row = schemas.get(db_id, row)
        indexes.setdefault(
            index_id,
            make_relational_index("spider", db_id, schema_row, sqlite_profiles),
        )
        position = counters[split]
        counters[split] += 1
        queries.append(
            {
                "id": f"spider:{split}:{position}",
                "query": row["question"],
                "sql": row.get("query") or row.get("SQL"),
                "source": "spider",
                "source_split": split,
                "source_query_id": position,
                "origin_index": index_id,
                "labels": {index_id: 2},
                "label_provenance": {index_id: "origin_gold"},
                "label_details": {},
                "difficulty": row.get("difficulty"),
            }
        )
    add_source_splits(indexes.values(), queries)
    return list(indexes.values()), queries


def load_bird(
    question_paths: Sequence[Path],
    schema_paths: Sequence[Path],
    sqlite_profile_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schemas = load_schema_rows(schema_paths)
    with sqlite_profile_path.open(encoding="utf-8") as handle:
        sqlite_profiles = json.load(handle)
    indexes: dict[str, dict[str, Any]] = {}
    queries = []
    counters: defaultdict[str, int] = defaultdict(int)

    for split, row in iter_input_rows(question_paths):
        db_id = str(row["db_id"])
        index_id = f"bird:{db_id}"
        schema_row = schemas.get(db_id, row)
        indexes.setdefault(
            index_id,
            make_relational_index("bird", db_id, schema_row, sqlite_profiles),
        )
        source_query_id = row.get("question_id")
        if source_query_id is None:
            source_query_id = counters[split]
        position = counters[split]
        counters[split] += 1
        queries.append(
            {
                "id": f"bird:{split}:{source_query_id}",
                "query": row["question"],
                "sql": row.get("SQL") or row.get("query"),
                "source": "bird",
                "source_split": split,
                "source_query_id": source_query_id,
                "origin_index": index_id,
                "labels": {index_id: 2},
                "label_provenance": {index_id: "origin_gold"},
                "label_details": {},
                "difficulty": row.get("difficulty"),
            }
        )
    add_source_splits(indexes.values(), queries)
    return list(indexes.values()), queries


def add_source_splits(
    indexes: Iterable[dict[str, Any]], queries: Sequence[dict[str, Any]]
) -> None:
    splits_by_index: defaultdict[str, set[str]] = defaultdict(set)
    for query in queries:
        splits_by_index[query["origin_index"]].add(query["source_split"])
    for index in indexes:
        index["source_splits"] = sorted(splits_by_index[index["index_id"]])


def read_feb4rag_resource_scores(path: Path) -> dict[tuple[str, str], int]:
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            query_id, _, resource, score = line.split()
            scores[(query_id, resource)] = int(score)
    return scores


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


def feb4rag_index_label(max_document_grade: int) -> int:
    if max_document_grade == 3:
        return 2
    if max_document_grade in (1, 2):
        return 1
    return 0


def load_feb4rag(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_root = root / "dataset"
    requests_path = dataset_root / "queries" / "requests.jsonl"
    mapping_path = dataset_root / "queries" / "rid_mapping.tsv"
    engines_path = dataset_root / "engines" / "engines.csv"
    rm_qrels_path = dataset_root / "qrels" / "BEIR-QRELS-RM.txt"
    rs_qrels_path = dataset_root / "qrels" / "BEIR-QRELS-RS.txt"
    profile_snapshot_path = root.parents[1] / "feb4rag" / "profile_snapshot.json"
    with profile_snapshot_path.open(encoding="utf-8") as handle:
        profile_snapshot = json.load(handle)

    request_rows = read_rows(requests_path)
    origins = {}
    with mapping_path.open(encoding="utf-8") as handle:
        for line in handle:
            query_id, resource, source_query_id = line.rstrip("\n").split("\t")
            origins[query_id] = (resource, source_query_id)

    indexes = []
    with engines_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            resource = row["name"]
            description = row.get("Description", "")
            profile_facts = profile_snapshot[resource]
            doc_count = profile_facts["doc_count"]
            vertical = row.get("vertical") or "unknown"
            task = row.get("Task") or "unknown"
            samples = []
            for sample in profile_facts["samples"]:
                enriched_sample = dict(sample)
                enriched_sample["__meta"] = {
                    "resource": resource,
                    "vertical": vertical,
                    "task": task,
                }
                samples.append(enriched_sample)
            feb4rag_mapping = {
                "_id": "keyword",
                "title": "text",
                "text": "text",
                "__meta.resource": "keyword",
                "__meta.vertical": "keyword",
                "__meta.task": "keyword",
            }
            for field in profile_facts["keyword_subfields"]:
                feb4rag_mapping[field] = "keyword"
            feb4rag_top_values = {
                "__meta.resource": [{"v": resource, "c": doc_count}],
                "__meta.vertical": [{"v": vertical, "c": doc_count}],
                "__meta.task": [{"v": task, "c": doc_count}],
            }
            feb4rag_top_values.update(profile_facts["field_top_values"])
            retrieval_text = " ".join(
                value
                for value in [
                    resource,
                    row.get("vertical", ""),
                    description,
                    row.get("Task", ""),
                    row.get("Objective (original)", ""),
                ]
                if value
            )
            indexes.append(
                {
                    "index_id": f"feb4rag:{resource}",
                    "source": "feb4rag",
                    "source_index_id": resource,
                    "kind": "document_collection",
                    "name": resource,
                    "description": description,
                    "retrieval_text": normalize_text(retrieval_text),
                    "profile": make_profile(
                        resource,
                        feb4rag_mapping,
                        field_top_values=feb4rag_top_values,
                        doc_count=doc_count,
                        samples=samples,
                        provenance={
                            "mapping_compressed": (
                                "beir_corpus_schema_plus_routing_metadata"
                            ),
                            "field_top_values": (
                                profile_facts["field_top_values_source"]
                                + "_plus_engine_metadata"
                            ),
                            "doc_count": profile_facts["doc_count_source"],
                            "samples": profile_facts["sample_source"],
                        },
                    ),
                    "schema": None,
                    "vertical": row.get("vertical"),
                    "task": row.get("Task"),
                    "source_splits": ["all"],
                }
            )

    document_grades = read_feb4rag_document_grades(
        rm_qrels_path, len(request_rows)
    )
    resource_scores = read_feb4rag_resource_scores(rs_qrels_path)
    queries = []
    for row in request_rows:
        query_id = str(row["_id"])
        origin_resource, source_query_id = origins.get(query_id, (None, None))
        labels = {}
        provenance = {}
        label_details = {}
        for resource in FEB4RAG_COLLECTIONS:
            index_id = f"feb4rag:{resource}"
            max_grade = document_grades[(query_id, resource)]
            labels[index_id] = feb4rag_index_label(max_grade)
            provenance[index_id] = "feb4rag_fused_top10_judgment"
            label_details[index_id] = {
                "max_document_grade": max_grade,
                "resource_score": resource_scores[(query_id, resource)],
            }
        queries.append(
            {
                "id": f"feb4rag:all:{query_id}",
                "query": row["text"],
                "sql": None,
                "source": "feb4rag",
                "source_split": "all",
                "source_query_id": source_query_id or query_id,
                "origin_index": (
                    f"feb4rag:{origin_resource}" if origin_resource else None
                ),
                "labels": labels,
                "label_provenance": provenance,
                "label_details": label_details,
                "difficulty": None,
            }
        )
    return indexes, queries


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_routing_splits(
    indexes: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
    seed: int,
) -> None:
    indexes_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for index in indexes:
        indexes_by_source[index["source"]].append(index)

    for source, source_indexes in indexes_by_source.items():
        if source in {"spider", "bird"}:
            train_indexes = [
                index for index in source_indexes if "train" in index["source_splits"]
            ]
            held_out_indexes = [
                index for index in source_indexes if index not in train_indexes
            ]
            for index in train_indexes:
                index["routing_split"] = "train"
            held_out_indexes.sort(
                key=lambda index: stable_seed(
                    seed, f"routing-split:{index['index_id']}"
                )
            )
            eval_count = len(held_out_indexes) // 2
            for index in held_out_indexes[:eval_count]:
                index["routing_split"] = "eval"
            for index in held_out_indexes[eval_count:]:
                index["routing_split"] = "test"
        elif source == "feb4rag":
            ordered = sorted(
                source_indexes,
                key=lambda index: stable_seed(
                    seed, f"routing-split:{index['index_id']}"
                ),
            )
            for position, index in enumerate(ordered):
                if position < 10:
                    index["routing_split"] = "train"
                elif position < 13:
                    index["routing_split"] = "eval"
                else:
                    index["routing_split"] = "test"
        else:
            raise ValueError(f"Unknown source: {source}")

    split_by_index = {
        index["index_id"]: index["routing_split"] for index in indexes
    }
    for query in queries:
        query["routing_split"] = split_by_index[query["origin_index"]]


def retrieve_ranked_ids(
    retriever: bm25s.BM25,
    query: str,
    source_index_ids: Sequence[str],
) -> tuple[list[str], dict[str, float]]:
    query_tokens = bm25s.tokenize([query], show_progress=False)
    results, scores = retriever.retrieve(
        query_tokens, corpus=list(source_index_ids), k=len(source_index_ids)
    )
    ranked = [str(index_id) for index_id in results[0]]
    score_by_id = {
        str(index_id): float(score)
        for index_id, score in zip(results[0], scores[0])
    }
    return ranked, score_by_id


def build_source_retriever(
    indexes: Sequence[dict[str, Any]],
) -> tuple[bm25s.BM25, list[str]]:
    index_ids = [row["index_id"] for row in indexes]
    corpus = [row["retrieval_text"] for row in indexes]
    corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    return retriever, index_ids


def choose_candidates(
    query: dict[str, Any],
    ranked_ids: Sequence[str],
    bm25_scores: dict[str, float],
    candidate_count: int,
    hard_negative_fraction: float,
    seed: int,
) -> dict[str, Any]:
    labels = query["labels"]
    eligible_ids = set(ranked_ids)
    full_gold = {
        index_id
        for index_id, label in labels.items()
        if label == 2 and index_id in eligible_ids
    }
    selected = list(sorted(full_gold))

    target_count = min(len(ranked_ids), max(candidate_count, len(selected)))
    remaining_slots = target_count - len(selected)
    hard_count = min(
        remaining_slots, math.ceil(remaining_slots * hard_negative_fraction)
    )

    bm25_candidates = [
        index_id for index_id in ranked_ids if index_id not in full_gold
    ]
    selected.extend(bm25_candidates[:hard_count])

    remaining_ids = [
        index_id for index_id in ranked_ids if index_id not in set(selected)
    ]
    rng = random.Random(stable_seed(seed, query["id"]))
    rng.shuffle(remaining_ids)
    selected.extend(remaining_ids[: target_count - len(selected)])
    rng.shuffle(selected)

    output_labels = [labels.get(index_id, -1) for index_id in selected]
    output = {
        "id": query["id"],
        "query": query["query"],
        "index": selected,
        "label": output_labels,
        "sql": query["sql"],
        "origin_index": query["origin_index"],
        "_source": query["source"],
        "_routing_split": query["routing_split"],
        "_has_full_gold": any(label == 2 for label in output_labels),
    }
    if query.get("difficulty") is not None:
        output["difficulty"] = query["difficulty"]
    return output


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            public_row = {
                key: value for key, value in row.items() if not key.startswith("_")
            }
            handle.write(
                json.dumps(public_row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_data_root = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_count < 1:
        raise ValueError("--candidate-count must be positive")
    if not 0 <= args.hard_negative_fraction <= 1:
        raise ValueError("--hard-negative-fraction must be between 0 and 1")

    raw_root = args.data_root / "raw"
    source_data = {
        "spider": load_spider(
            [
                raw_root / "spider" / "train.parquet",
                raw_root / "spider" / "validation.parquet",
            ],
            [raw_root / "spider" / "spider_schema_rows_v2.json"],
            args.data_root / "spider" / "sqlite_profile_snapshot.json",
        ),
        "bird": load_bird(
            [
                raw_root / "bird" / "train_with_schema.json",
                raw_root / "bird" / "dev_with_schema.json",
            ],
            [],
            args.data_root / "bird" / "sqlite_profile_snapshot.json",
        ),
        "feb4rag": load_feb4rag(raw_root / "feb4rag"),
    }
    all_indexes = [
        index for indexes, _ in source_data.values() for index in indexes
    ]
    all_queries = [
        query for _, queries in source_data.values() for query in queries
    ]
    assign_routing_splits(all_indexes, all_queries, args.seed)

    indexes_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for index in all_indexes:
        indexes_by_source[index["source"]].append(index)

    indexes_by_pool: defaultdict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for index in all_indexes:
        indexes_by_pool[
            (index["source"], index["routing_split"])
        ].append(index)

    retrievers = {
        pool: build_source_retriever(indexes)
        for pool, indexes in indexes_by_pool.items()
    }

    samples = []
    for query in all_queries:
        retriever, source_index_ids = retrievers[
            (query["source"], query["routing_split"])
        ]
        ranked_ids, bm25_scores = retrieve_ranked_ids(
            retriever, query["query"], source_index_ids
        )
        samples.append(
            choose_candidates(
                query,
                ranked_ids,
                bm25_scores,
                args.candidate_count,
                args.hard_negative_fraction,
                args.seed,
            )
        )

    summaries = {}
    for source in sorted(indexes_by_source):
        output_dir = args.data_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        source_indexes = sorted(
            indexes_by_source[source], key=lambda row: row["index_id"]
        )
        source_samples = [row for row in samples if row["_source"] == source]
        write_jsonl(output_dir / "indexes.jsonl", source_indexes)
        for split in ("train", "eval", "test"):
            split_samples = [
                row for row in source_samples if row["_routing_split"] == split
            ]
            write_jsonl(output_dir / f"{split}.jsonl", split_samples)

        summary = {
            "indexes": len(source_indexes),
            "samples": len(source_samples),
            "candidate_count": args.candidate_count,
            "hard_negative_fraction": args.hard_negative_fraction,
            "seed": args.seed,
            "split_indexes": {
                split: sum(
                    index["routing_split"] == split for index in source_indexes
                )
                for split in ("train", "eval", "test")
            },
            "split_samples": {
                split: sum(
                    row["_routing_split"] == split for row in source_samples
                )
                for split in ("train", "eval", "test")
            },
            "samples_with_full_gold": sum(
                row["_has_full_gold"] for row in source_samples
            ),
            "label_counts": {
                str(label): sum(
                    candidate_label == label
                    for row in source_samples
                    for candidate_label in row["label"]
                )
                for label in (-1, 0, 1, 2)
            },
        }
        with (output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        summaries[source] = summary
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
