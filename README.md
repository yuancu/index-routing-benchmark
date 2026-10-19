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
│   │   ├── sqlite_profile_snapshot.json
│   │   ├── train.jsonl
│   │   ├── eval.jsonl
│   │   ├── test.jsonl
│   │   └── summary.json
│   ├── bird/
│   │   └── sqlite_profile_snapshot.json
│   └── feb4rag/
│       └── profile_snapshot.json
├── scripts/
│   ├── build_dataset.py
│   ├── build_sqlite_profile_snapshot.py
│   └── fetch_feb4rag_profile_snapshot.py
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
  "profile": {
    "index": "department_management",
    "mapping_compressed": {
      "__meta.table": "keyword",
      "__meta.column_type": "keyword",
      "department.Department_ID": "long",
      "department.Name": "text"
    },
    "field_top_values": {
      "__meta.table": [{"v": "department", "c": 1}],
      "__meta.column_type": [
        {"v": "number", "c": 1},
        {"v": "text", "c": 1}
      ]
    },
    "time_ranges": {},
    "doc_count": null,
    "samples": [],
    "retrieval_capabilities": {
      "has_vector_field": false,
      "vector_fields": [],
      "text_fields": ["department.Name"],
      "keyword_fields": ["__meta.table", "__meta.column_type"],
      "filter_fields": [
        "__meta.table",
        "__meta.column_type",
        "department.Department_ID"
      ],
      "date_fields": []
    },
    "default_pipelines": {},
    "profile_provenance": {
      "mapping_compressed": "source_database_schema_plus_routing_metadata",
      "field_top_values": "source_schema_low_cardinality_metadata",
      "doc_count": "source_schema_table_count",
      "samples": "source_schema_richest_table"
    }
  },
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
column names, and any available column descriptions. FeB4RAG uses each
engine's released name, vertical, description, task, and objective. No
LLM-generated index descriptions are used in the current dataset.

### Simulated index profiles

Every index record contains a `profile` modeled after
`OpenSearchIndexProfileBuilder` in `AISearchDataPlaneMetadataService`. The
service serializes this profile as a JSON string; `indexes.jsonl` stores the
parsed JSON object so benchmark consumers do not need to decode it twice. The
profile shape is:

```json
{
  "index": "index name",
  "mapping_compressed": {"field.path": "field_type"},
  "field_top_values": {},
  "time_ranges": {},
  "doc_count": null,
  "samples": [],
  "retrieval_capabilities": {
    "has_vector_field": false,
    "vector_fields": [],
    "text_fields": [],
    "keyword_fields": [],
    "filter_fields": [],
    "date_fields": []
  },
  "default_pipelines": {},
  "profile_provenance": {}
}
```

The production service obtains the mapping from OpenSearch `_mapping`, keeps
top ten values for keyword fields with cardinality below 1,000, computes
minimum and maximum values for date fields, reads `_count`, selects the most
populated document among three sample candidates, and reads default ingest
and search pipelines.

This benchmark does not retain live OpenSearch indexes or complete copies of
the large source collections. `profile_provenance` records how every
content-dependent field was filled:

- Spider and BIRD flatten each database schema to `table.column` paths. SQL
  integer/number fields map to `long`, real/float/decimal fields to `double`,
  date/time fields to `date`, boolean fields to `boolean`, and other fields
  to `text`. Their profile treats every table schema as one metadata document:
  `doc_count` is the number of tables and `samples` contains the table with
  the most columns. The sample is explicitly marked
  `"_sample_type": "table_schema"` and must not be interpreted as a database
  row.
- Their simulated index also contains low-cardinality `__meta.table` and
  `__meta.column_type` keyword fields. `field_top_values` lists at most ten
  table names and source column types with their table-level frequencies.
- The checked-in SQLite profile snapshots add real database observations.
  Spider covers 159 non-empty databases; its `music_2` SQLite file has no
  rows and therefore keeps the schema fallback. BIRD covers 65 databases:
  all 11 mini-dev databases and 54 training databases whose individual
  SQLite files are at most 200 MiB. The remaining 15 large BIRD training
  databases keep the schema fallback.
- For each available SQLite database, `doc_count` is the sum of rows across
  its tables. `samples` is the most populated row among the first three rows
  of each table. Low-cardinality text columns gain a simulated
  `Table.Column.keyword` mapping, and their actual Top-10 values are stored in
  `field_top_values`.
- FeB4RAG resources use the common BEIR document mapping: `_id` is `keyword`,
  while `title` and `text` are `text`. For publicly hosted BEIR collections,
  exact corpus row counts are fetched through the Hugging Face dataset server
  without downloading the full corpus. Twelve collections also provide one
  real document sample through its first-row API.
- Each FeB4RAG document sample is enriched with `__meta.resource`,
  `__meta.vertical`, and `__meta.task`. These are deterministic keyword fields
  derived from `engines.csv`; their top-value count equals the corpus
  `doc_count`, modeling metadata copied to every document in that resource.
- FeB4RAG also exposes a non-metadata keyword subfield. The profile uses
  `title.keyword`, or `text.keyword` when titles are empty, and computes its
  values over the rows returned by the dataset server's first-row endpoint.
  These counts are sample frequencies, not corpus-wide frequencies, and
  `profile_provenance.field_top_values` records that distinction.
- Signal-1M, TREC-NEWS, and Robust04 cannot be redistributed by BEIR. Their
  `doc_count` uses BEIR's published rounded corpus size, and their sample is a
  clearly identified fallback built from FeB4RAG's released engine metadata.
- Webis-Touche2020 has an exact fetched count, but its first-row endpoint does
  not return a sample, so it uses the same engine-metadata fallback.
- `retrieval_capabilities` is derived deterministically from that mapping.
- `time_ranges` and `default_pipelines` remain empty because the retained
  metadata cannot support those observations.

`data/feb4rag/profile_snapshot.json` stores the small fetched result so normal
dataset builds are offline and reproducible. Refresh it explicitly with
`scripts/fetch_feb4rag_profile_snapshot.py`; the script downloads metadata and
one document row per public collection, not the full corpora.

These are simulated profiles for routing experiments; they are not profiles
released by Spider, BIRD, FeB4RAG, or returned from a live metadata service.

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
.venv/bin/python scripts/fetch_feb4rag_profile_snapshot.py
.venv/bin/python scripts/build_dataset.py
```

The committed SQLite snapshots make the normal build independent of database
downloads. To regenerate them from downloaded ZIP archives:

```bash
.venv/bin/python scripts/build_sqlite_profile_snapshot.py \
  --archive /path/to/spider.zip \
  --output data/spider/sqlite_profile_snapshot.json

.venv/bin/python scripts/build_sqlite_profile_snapshot.py \
  --archive /path/to/bird-mini-dev-sqlite.zip \
  --output data/bird/sqlite_profile_snapshot.json

.venv/bin/python scripts/build_sqlite_profile_snapshot.py \
  --sqlite /path/to/address.sqlite /path/to/airline.sqlite \
  --output data/bird/sqlite_profile_snapshot.json \
  --merge
```

Optional parameters:

```bash
.venv/bin/python scripts/build_dataset.py \
  --candidate-count 8 \
  --hard-negative-fraction 0.75 \
  --seed 17
```
