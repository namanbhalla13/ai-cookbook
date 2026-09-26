"""
Explore the locally cached FiQA dataset.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "fiqa"

# --------------------------------------------------------------
# 1. Load datasets
# --------------------------------------------------------------

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
queries = pd.read_parquet(DATA_DIR / "queries.parquet")
qrels = pd.read_parquet(DATA_DIR / "qrels.parquet")


# --------------------------------------------------------------
# 2. Normalize ID types
# --------------------------------------------------------------

corpus["_id"] = corpus["_id"].astype(str)
queries["_id"] = queries["_id"].astype(str)

qrels["query-id"] = qrels["query-id"].astype(str)
qrels["corpus-id"] = qrels["corpus-id"].astype(str)


# --------------------------------------------------------------
# 3. Basic overview
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SHAPES")
print("=" * 70)

print(f"Corpus : {corpus.shape}")
print(f"Queries: {queries.shape}")
print(f"Qrels  : {qrels.shape}")


# --------------------------------------------------------------
# 4. Look at one row from each dataset
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("EXAMPLE QUERY")
print("=" * 70)

example_query = queries.iloc[0]
print(example_query)


print("\n" + "=" * 70)
print("EXAMPLE CORPUS")
print("=" * 70)

example_corpus = corpus.iloc[0]
print(example_corpus)


print("\n" + "=" * 70)
print("EXAMPLE QREL")
print("=" * 70)

example_qrel = qrels.iloc[0]
print(example_qrel)


# --------------------------------------------------------------
# 5. Pick a query that definitely exists in qrels
# --------------------------------------------------------------

query_id = qrels.iloc[0]["query-id"]

print("\nSelected query ID:", query_id)


# Find corresponding query
matching_queries = queries[
    queries["_id"] == query_id
]

if matching_queries.empty:
    raise ValueError(f"Query ID {query_id} not found in queries")

example_query = matching_queries.iloc[0]


print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print("Query ID:", query_id)
print("Query:", example_query["text"])


# --------------------------------------------------------------
# 6. Find all ground-truth relevant docs
# --------------------------------------------------------------

relevant = qrels[
    qrels["query-id"] == query_id
]

print("\n" + "=" * 70)
print("QRELS")
print("=" * 70)

print(relevant.to_string(index=False))


# --------------------------------------------------------------
# 7. Print corresponding corpus documents
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("RELEVANT DOCUMENTS")
print("=" * 70)

for _, row in relevant.iterrows():

    doc_id = row["corpus-id"]

    matching_docs = corpus[
        corpus["_id"] == doc_id
    ]

    if matching_docs.empty:
        print(f"\nWARNING: document {doc_id} not found in corpus")
        continue

    doc = matching_docs.iloc[0]

    print(f"\nDocument ID: {doc_id}")
    print(f"Relevance score: {row['score']}")
    print(f"Title: {doc['title']}")
    print(f"\nText:\n{doc['text']}")
    print("-" * 70)
