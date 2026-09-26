"""
Create one JSON file containing the full FiQA corpus.

This full corpus will be used by the retriever.
Do NOT filter it using qrels.
"""

from pathlib import Path
import pandas as pd
import json


# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "fiqa"

CORPUS_PATH = DATA_DIR / "corpus.parquet"

OUTPUT_PATH = DATA_DIR / "full_corpus.json"


# --------------------------------------------------------------
# Load full corpus
# --------------------------------------------------------------

corpus = pd.read_parquet(CORPUS_PATH)

corpus["_id"] = corpus["_id"].astype(str)


print(f"Total corpus documents: {len(corpus)}")


# --------------------------------------------------------------
# Convert to JSON-friendly structure
# --------------------------------------------------------------

full_corpus = []

for _, row in corpus.iterrows():

    full_corpus.append(
        {
            "corpus_id": row["_id"],
            "title": row["title"],
            "text": row["text"],
        }
    )


# --------------------------------------------------------------
# Save
# --------------------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        full_corpus,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 80)
print("FULL CORPUS CREATED")
print("=" * 80)

print(f"Documents: {len(full_corpus)}")
print(f"Saved to: {OUTPUT_PATH}")
