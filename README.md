# Index Routing Benchmark

This directory converts the original Spider, BIRD, and FeB4RAG data into
source-specific index-routing datasets.

## Layout

```text
index_routing_benchmark/
├── data/
│   ├── raw/
│   │   ├── spider/
│   │   ├── bird/
│   │   └── feb4rag/
│   ├── spider/
│   │   ├── indexes.jsonl
│   │   ├── train.jsonl
│   │   ├── eval.jsonl
│   │   ├── test.jsonl
│   │   └── summary.json
│   ├── bird/
│   └── feb4rag/
├── scripts/
│   └── build_dataset.py
└── requirements.txt
```

`data/raw/` retains the downloaded source files without modifying them.
Generated datasets are grouped by source under `data/<source>/`. Dataset
versions are managed by Git, so generated paths do not contain a version
number.

## Raw inputs

| Source | Files used |
|---|---|
| Spider | `train.parquet`, `validation.parquet`, `spider_schema_rows_v2.json` |
| BIRD | `train_with_schema.json`, `dev_with_schema.json` |
| FeB4RAG | `requests.jsonl`, `rid_mapping.tsv`, `engines.csv`, `BEIR-QRELS-RM.txt`, `BEIR-QRELS-RS.txt` |

The Spider question files are the Hugging Face representation of Spider. The
schema file is a public schema-only representation. The BIRD files are public
representations with each database schema embedded in its question records.
FeB4RAG retains the released query, engine, and qrels files under its original
`dataset/` subdirectories.

The SQLite database contents are not required by the current transformation.
They will be needed later if Spider/BIRD candidate indexes are judged through
query execution.

## Routing sample

Generated query records contain:

```json
{
  "id": "spider:train:0",
  "query": "How many department heads are older than 56?",
  "index": ["spider:department_management", "spider:college_2"],
  "label": [2, -1],
  "sql": "SELECT count(*) FROM head WHERE age > 56",
  "origin_index": "spider:department_management"
}
```

`sql` is `null` for FeB4RAG.

Labels mean:

| Label | Meaning |
|---:|---|
| `2` | The index has evidence that can fully answer the query |
| `1` | The index contains relevant information but is insufficient alone |
| `0` | The index was judged unable to answer |
| `-1` | The index has not been judged |

Spider and BIRD only identify the database from which a query originated.
Their other candidate indexes remain `-1`; treating them as benchmark
negatives would create false negatives. FeB4RAG's released Top-10 document
judgments are converted as follows:

| Maximum document grade in an index | Index label |
|---:|---:|
| `3` | `2` |
| `1` or `2` | `1` |
| `0` | `0` |

## Index catalog

Each source has its own `indexes.jsonl`. Schemas and descriptions are stored
once instead of being repeated in every query:

```json
{
  "index_id": "spider:department_management",
  "source": "spider",
  "source_index_id": "department_management",
  "routing_split": "train",
  "kind": "relational_database",
  "name": "department management",
  "description": "Database with tables: department, head, management.",
  "retrieval_text": "department management department head management ...",
  "schema": {}
}
```

Descriptions and BM25 retrieval text are derived only from index metadata and
schemas, never from held-out questions.

For Spider and BIRD, the description is produced deterministically:

```text
Database with tables: <table 1>, <table 2>, ...
```

The BM25 document contains the index name, generated description, table names,
column names, column types, and any available column descriptions. FeB4RAG
uses each engine's released name, vertical, description, task, and objective.
No LLM-generated index descriptions are used in the current dataset.

## Splits

The current benchmark contains an unseen-index evaluation track. Index
partitions are:

- Spider's original training indexes remain `train`; its 20 development
  indexes are deterministically divided into 10 `eval` and 10 `test` indexes.
- BIRD's 69 training indexes remain `train`; its 11 development indexes are
  deterministically divided between `eval` and `test`.
- FeB4RAG's 16 resources are deterministically divided into 10 `train`, 3
  `eval`, and 3 `test` resources. Its queries follow the split assigned to
  their origin resource in `rid_mapping.tsv`.

Queries from train indexes form `train.jsonl`. Queries from eval/test index
partitions form `eval.jsonl` and `test.jsonl`. BM25 candidate retrieval is
restricted to the corresponding index partition, so eval/test index metadata
never appears in training samples.

A separate seen-index evaluation track is intentionally not generated yet.

## Candidate generation

For each query, the builder:

1. searches index `retrieval_text` with an in-memory `bm25s` index;
2. forces all full gold indexes from that routing split into the candidate set;
3. takes 75% of remaining slots from BM25-ranked hard candidates;
4. fills the remaining slots with deterministic random candidates;
5. shuffles the final candidates so BM25 rank is not exposed by position.

Generated candidate lists are materialized in JSONL and do not change during
evaluation.

Candidate selection is deterministic for a given seed. Index partitions and
per-query random candidates are ordered using SHA-256-derived seeds. If a
split contains fewer than the requested number of candidates, all indexes in
that split are used.

## Complete transformation

`scripts/build_dataset.py` performs these steps:

1. Load the raw question records and database/resource metadata.
2. Deduplicate schema metadata into one `indexes.jsonl` record per index.
3. Generate index descriptions and BM25 retrieval text without using queries.
4. Create labels:
   - Spider/BIRD origin index: `2`;
   - Spider/BIRD other candidates: `-1` because they are unjudged;
   - FeB4RAG: take the maximum released document grade among an index's
     Top-10 results, then map `3 → 2`, `1/2 → 1`, and `0 → 0`.
5. Partition indexes into train, eval, and test using seed `17`.
6. Build a separate in-memory BM25 index for every source and split.
7. Retrieve hard candidates, force full-gold indexes into the set, add random
   candidates, and shuffle their final order.
8. Write `train.jsonl`, `eval.jsonl`, `test.jsonl`, `indexes.jsonl`, and
   `summary.json` independently for each source.

The `index` and `label` arrays are positional: `label[i]` describes
`index[i]`. The build script validates candidate availability through the
source-specific index catalog while constructing each record.

## Build

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -r requirements-parquet.txt
.venv/bin/python scripts/build_dataset.py
```

Optional parameters:

```bash
.venv/bin/python scripts/build_dataset.py \
  --candidate-count 8 \
  --hard-negative-fraction 0.75 \
  --seed 17
```
