"""
Create 50-query FiQA gold dataset as JSON.

Each query contains:
- query_id
- question
- relevant_docs
    - corpus_id
    - score
    - title
    - text
"""

from pathlib import Path
import pandas as pd
import json


# --------------------------------------------------------------
# 1. Paths
# --------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "fiqa"

CORPUS_PATH = DATA_DIR / "corpus.parquet"
QUERIES_PATH = DATA_DIR / "queries.parquet"
QRELS_PATH = DATA_DIR / "qrels.parquet"

OUTPUT_PATH = DATA_DIR / "gold_dataset_50.json"


# --------------------------------------------------------------
# 2. Load datasets
# --------------------------------------------------------------

corpus = pd.read_parquet(CORPUS_PATH)
queries = pd.read_parquet(QUERIES_PATH)
qrels = pd.read_parquet(QRELS_PATH)


# --------------------------------------------------------------
# 3. Normalize IDs
# --------------------------------------------------------------

corpus["_id"] = corpus["_id"].astype(str)
queries["_id"] = queries["_id"].astype(str)

qrels["query-id"] = qrels["query-id"].astype(str)
qrels["corpus-id"] = qrels["corpus-id"].astype(str)


# --------------------------------------------------------------
# 4. Only use queries that have gold documents
# --------------------------------------------------------------

gold_query_ids = qrels["query-id"].unique()

valid_queries = queries[
    queries["_id"].isin(gold_query_ids)
].copy()

print(f"Queries with gold documents: {len(valid_queries)}")


# --------------------------------------------------------------
# 5. Select 50 unique queries
# --------------------------------------------------------------

N = 50

sampled_queries = valid_queries.sample(
    n=min(N, len(valid_queries)),
    random_state=42
)


# --------------------------------------------------------------
# 6. Make corpus lookup
# --------------------------------------------------------------
# Much faster than filtering the entire dataframe repeatedly

corpus_lookup = corpus.set_index("_id").to_dict("index")


# --------------------------------------------------------------
# 7. Build gold dataset
# --------------------------------------------------------------

gold_dataset = []

for _, query_row in sampled_queries.iterrows():

    query_id = query_row["_id"]
    question = query_row["text"]

    # Get all gold documents for this query
    relevant_qrels = qrels[
        qrels["query-id"] == query_id
    ]

    relevant_docs = []

    for _, qrel_row in relevant_qrels.iterrows():

        corpus_id = qrel_row["corpus-id"]
        score = int(qrel_row["score"])

        # Make sure document exists
        if corpus_id not in corpus_lookup:
            print(f"WARNING: Corpus ID {corpus_id} not found")
            continue

        document = corpus_lookup[corpus_id]

        relevant_docs.append(
            {
                "corpus_id": corpus_id,
                "score": score,
                "title": document["title"],
                "text": document["text"],
            }
        )

    gold_dataset.append(
        {
            "query_id": query_id,
            "question": question,
            "relevant_docs": relevant_docs,
        }
    )


# --------------------------------------------------------------
# 8. Save JSON
# --------------------------------------------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    json.dump(
        gold_dataset,
        f,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------------------
# 9. Summary
# --------------------------------------------------------------

print("\n" + "=" * 80)
print("GOLD DATASET CREATED")
print("=" * 80)

print(f"Queries: {len(gold_dataset)}")

total_relevant_docs = sum(
    len(item["relevant_docs"])
    for item in gold_dataset
)

print(f"Total relevant documents: {total_relevant_docs}")

print(f"\nSaved to:")
print(OUTPUT_PATH)


# --------------------------------------------------------------
# 10. Print first example
# --------------------------------------------------------------

print("\n" + "=" * 80)
print("EXAMPLE")
print("=" * 80)

print(
    json.dumps(
        gold_dataset[0],
        indent=2,
        ensure_ascii=False,
    )
)
