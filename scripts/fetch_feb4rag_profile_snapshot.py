#!/usr/bin/env python3
"""Fetch small BEIR profile facts without downloading full document corpora."""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


COLLECTIONS = [
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

# BEIR publishes rounded corpus sizes for collections it cannot redistribute.
RESTRICTED_CORPUS_COUNTS = {
    "signal1m": 2_860_000,
    "trec-news": 595_000,
    "robust04": 528_000,
}

MAX_STRING_CHARS = 150
TRUNCATED_SUFFIX = "...[truncated]"


def get_json(endpoint: str, **params: str) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/{endpoint}?{query}",
        headers={"User-Agent": "index-routing-benchmark-profile-fetcher"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        return value[:MAX_STRING_CHARS] + TRUNCATED_SUFFIX
    if isinstance(value, dict):
        return {key: clip(item) for key, item in value.items()}
    if isinstance(value, list):
        clipped = [clip(item) for item in value[:3]]
        if len(value) > 3:
            clipped.append(f"...[{len(value) - 3} more]")
        return clipped
    return value


def load_engine_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def metadata_sample(resource: str, engine: dict[str, str]) -> dict[str, str]:
    return clip(
        {
            "_id": f"metadata:{resource}",
            "title": engine.get("Task", ""),
            "text": engine.get("Description", ""),
        }
    )


def public_profile(
    resource: str, engine: dict[str, str]
) -> dict[str, Any]:
    dataset = f"BeIR/{resource}"
    size = get_json("size", dataset=dataset)["size"]
    corpus_size = next(
        config["num_rows"]
        for config in size["configs"]
        if config["config"] == "corpus"
    )
    try:
        first_rows = get_json(
            "first-rows", dataset=dataset, config="corpus", split="corpus"
        )
        rows = [item["row"] for item in first_rows["rows"]]
        sample = clip(rows[0])
        sample_source = "huggingface_beir_first_rows_api"
        observed_field = "title"
        observed_values = [
            str(row.get(observed_field, "")).strip()
            for row in rows
            if str(row.get(observed_field, "")).strip()
        ]
        if not observed_values:
            observed_field = "text"
            observed_values = [
                str(row.get(observed_field, "")).strip()
                for row in rows
                if str(row.get(observed_field, "")).strip()
            ]
        counts = Counter(observed_values)
        keyword_field = f"{observed_field}.keyword"
        field_top_values = {
            keyword_field: [
                {"v": clip(value), "c": count}
                for value, count in sorted(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ]
        }
        keyword_subfields = [keyword_field]
        top_values_source = "huggingface_beir_first_rows_sample_frequency"
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError):
        sample = metadata_sample(resource, engine)
        sample_source = "feb4rag_engine_metadata_fallback"
        keyword_subfields = ["title.keyword"]
        field_top_values = {
            "title.keyword": [{"v": sample["title"], "c": 1}]
        }
        top_values_source = "feb4rag_engine_metadata_fallback"
    return {
        "doc_count": corpus_size,
        "samples": [sample],
        "keyword_subfields": keyword_subfields,
        "field_top_values": field_top_values,
        "doc_count_source": "huggingface_beir_size_api",
        "sample_source": sample_source,
        "field_top_values_source": top_values_source,
    }


def restricted_profile(
    resource: str, engine: dict[str, str]
) -> dict[str, Any]:
    sample = metadata_sample(resource, engine)
    return {
        "doc_count": RESTRICTED_CORPUS_COUNTS[resource],
        "samples": [sample],
        "keyword_subfields": ["title.keyword"],
        "field_top_values": {
            "title.keyword": [{"v": sample["title"], "c": 1}]
        },
        "doc_count_source": "beir_published_rounded_count",
        "sample_source": "feb4rag_engine_metadata_fallback",
        "field_top_values_source": "feb4rag_engine_metadata_fallback",
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    engines_path = (
        project_root
        / "data"
        / "raw"
        / "feb4rag"
        / "dataset"
        / "engines"
        / "engines.csv"
    )
    output_path = (
        project_root
        / "data"
        / "feb4rag"
        / "profile_snapshot.json"
    )
    engines = load_engine_metadata(engines_path)
    profiles = {}
    for resource in COLLECTIONS:
        try:
            profiles[resource] = public_profile(resource, engines[resource])
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError):
            if resource not in RESTRICTED_CORPUS_COUNTS:
                raise
            profiles[resource] = restricted_profile(resource, engines[resource])
        print(resource, profiles[resource]["doc_count_source"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(profiles, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
