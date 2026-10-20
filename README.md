# Index Routing Benchmark

Pairwise index-routing data: given a query and one index's metadata, is the
index needed? Each episode lists every candidate index of the query's pool,
so the answer may be zero, one, or several indexes.

| Source | Index | Candidates per episode | Positive label |
|---|---|---|---|
| Spider | one table | all tables of the query's database | reference SQL references the table |
| BIRD | one table | all tables of the query's database | reference SQL references the table |
| FeB4RAG | one search resource | all resources in the query's split | best released Top-10 document grade ≥ 2 |

## Layout

```text
index_routing_benchmark/
├── data/
│   ├── raw/                      # downloaded inputs, not in Git
│   ├── spider/  bird/  feb4rag/
│   │   ├── indexes.jsonl
│   │   ├── train.jsonl  eval.jsonl  test.jsonl
│   │   ├── summary.json
│   │   ├── parse_failures.jsonl  # Spider/BIRD only
│   │   └── sqlite_profile_snapshot.json | profile_snapshot.json
├── scripts/
│   ├── build_dataset.py
│   ├── build_sqlite_profile_snapshot.py
│   └── fetch_feb4rag_profile_snapshot.py
└── table_routing_design.md
```

## Episodes

```json
{
  "id": "spider:train:6",
  "query": "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?",
  "index": [
    "spider:department_management:management",
    "spider:department_management:head",
    "spider:department_management:department"
  ],
  "label": [1, 1, 1],
  "sql": "SELECT DISTINCT T1.creation FROM department AS T1 JOIN management AS T2 ...",
  "database_id": "spider:department_management"
}
```

`label[i]` describes `index[i]`; candidate order is shuffled with a fixed
seed. Expand an episode into pairs and join `index` against `indexes.jsonl`.

Audit-only fields, not model input: `sql` and `database_id` (Spider/BIRD);
`max_document_grade` (the raw 0–3 grade per candidate) and `origin_index`
(FeB4RAG).

Spider/BIRD labels mean *the reference SQL uses this table*
(`reference_table_usage`); they do not claim that an unused table could never
support another correct query. Table references are extracted with `sqlglot`
(SQLite dialect) across JOINs, subqueries and set operations; aliases, CTE
names and self-joins are resolved to physical tables. Queries whose SQL
cannot be parsed, or that reference a table missing from the database, are
excluded and listed in `parse_failures.jsonl`. They are never turned into
negatives. Spider has 1 such query. BIRD has 142, all in train: its
`retail_world` and `works_cycles` SQL references tables that are absent from
the shipped SQLite files.

## Index catalog

Spider/BIRD table index:

```json
{
  "index_id": "spider:department_management:management",
  "database_id": "spider:department_management",
  "source": "spider",
  "source_index_id": "management",
  "routing_split": "train",
  "kind": "relational_table",
  "name": "management",
  "description": "Table management with columns: department_ID, head_ID, temporary_acting.",
  "schema": {
    "columns": [{"name": "department_ID", "type": "number"}, "..."],
    "primary_key": ["department_ID"],
    "foreign_keys": [
      {
        "columns": ["head_ID"],
        "referenced_index": "spider:department_management:head",
        "referenced_columns": ["head_ID"]
      }
    ]
  },
  "profile": {
    "index": "spider:department_management:management",
    "mapping_compressed": {
      "department_ID": "long",
      "head_ID": "long",
      "temporary_acting": "text",
      "temporary_acting.keyword": "keyword"
    },
    "field_top_values": {
      "temporary_acting.keyword": [{"v": "Yes", "c": 3}, {"v": "No", "c": 2}]
    },
    "time_ranges": {},
    "doc_count": 5,
    "samples": [{"department_ID": 2, "head_ID": 5, "temporary_acting": "Yes"}],
    "retrieval_capabilities": {"...": "derived from mapping_compressed"},
    "default_pipelines": {},
    "profile_provenance": {"doc_count": "sqlite_table_row_count", "...": "..."}
  }
}
```

Columns may also carry `natural_name` (from the official `tables.json`) and,
for BIRD, `description` (static column and value descriptions from
`database_description/*.csv`). BIRD's query-specific `evidence` is never
used. SQLite's internal `sqlite_sequence` table is not an index.

The `profile` imitates `OpenSearchIndexProfileBuilder` in
`AISearchDataPlaneMetadataService`, computed per table from the real SQLite
data:

- `mapping_compressed`: SQL integer/number → `long`, real/float/decimal →
  `double`, date/time → `date`, boolean → `boolean`, otherwise `text`. Text
  columns with fewer than 1,000 distinct values gain a `<column>.keyword`
  field.
- `field_top_values`: Top-10 values of those keyword fields.
- `doc_count`: exact row count. It is 0 for the 13 Spider tables (in
  `music_2`, `formula_1` and `sakila_1`) and the 2 BIRD tables that are empty
  in the release.
- `samples`: the most populated row among the first three. Strings are
  truncated at 150 characters.
- `time_ranges` and `default_pipelines` are empty.

FeB4RAG resource indexes use the BEIR document mapping (`_id`, `title`,
`text`) plus `__meta.resource/vertical/task` keyword fields from
`engines.csv`. Row counts and one sample document come from the Hugging Face
dataset server (`profile_snapshot.json`). For Signal-1M, TREC-NEWS and
Robust04, which BEIR cannot redistribute, the rounded published size and an
engine-metadata sample are used instead. `profile_provenance` records the
origin of each field.

## Splits

Splits are by index pool, so eval/test indexes never appear in training.

- Spider/BIRD: databases from the original training set are `train`. The
  original development databases are split evenly into `eval` and `test`
  (seed `17`). All tables of a database share its split.
- FeB4RAG: the 16 resources are split 10 / 3 / 3. A query follows its origin
  resource's split, and its candidates are all resources in that split.

| Source | Split | Indexes | Episodes | Pairs | Positive pairs |
|---|---|---:|---:|---:|---:|
| Spider | train | 735 | 6,999 | 42,110 | 10,874 |
| Spider | eval | 40 | 434 | 1,944 | 658 |
| Spider | test | 40 | 600 | 2,613 | 907 |
| BIRD | train | 522 | 9,286 | 109,258 | 18,636 |
| BIRD | eval | 21 | 652 | 2,802 | 1,171 |
| BIRD | test | 54 | 882 | 8,055 | 1,807 |
| FeB4RAG | train | 10 | 490 | 4,900 | 821 |
| FeB4RAG | eval | 3 | 150 | 450 | 163 |
| FeB4RAG | test | 3 | 150 | 450 | 211 |

`summary.json` also reports positives per episode. Spider/BIRD episodes
always contain at least one positive. FeB4RAG episodes may contain none
(113 / 66 / 33 per split).

## Raw inputs

| Source | Files under `data/raw/<source>/` |
|---|---|
| Spider | `train.parquet`, `validation.parquet` (Hugging Face `xlangai/spider`), `tables.json` (official release) |
| BIRD | `train_with_schema.json`, `dev_with_schema.json`, `train_tables.json`, `dev_tables.json`, `database_description/<db>/<table>.csv` (official `train.zip` / `dev.zip`) |
| FeB4RAG | `dataset/queries/{requests.jsonl,rid_mapping.tsv}`, `dataset/engines/engines.csv`, `dataset/qrels/BEIR-QRELS-RM.txt` |

## Build

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
.venv/bin/python scripts/build_dataset.py            # all sources, offline
.venv/bin/python scripts/build_dataset.py --sources bird
```

The committed profile snapshots keep normal builds offline. To refresh them:

```bash
# Spider: official spider_data.zip → spider_data/database/<db>/<db>.sqlite
.venv/bin/python scripts/build_sqlite_profile_snapshot.py \
  --sqlite /path/to/spider_data/database/*/*.sqlite \
  --output data/spider/sqlite_profile_snapshot.json

# BIRD: https://bird-bench.oss-cn-beijing.aliyuncs.com/{train,dev}.zip contain
# nested *_databases.zip; the 9 GB train one needs Python zipfile (unzip
# rejects it as a zip bomb).
.venv/bin/python scripts/build_sqlite_profile_snapshot.py \
  --sqlite /path/to/train_databases/*/*.sqlite /path/to/dev_databases/*/*.sqlite \
  --output data/bird/sqlite_profile_snapshot.json

.venv/bin/python scripts/fetch_feb4rag_profile_snapshot.py
```
