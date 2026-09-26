from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi


# --------------------------------------------------------------
# 1. Paths
# --------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "fiqa"
CORPUS_PATH = DATA_DIR / "full_corpus.json"

TOP_K = 10


# --------------------------------------------------------------
# 2. Simple tokenizer
# --------------------------------------------------------------

def tokenize(text: str):
    """
    Lowercase + simple word tokenization.
    """
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------------------
# 3. Load corpus
# --------------------------------------------------------------

print("Loading corpus...")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Loaded {len(corpus)} documents.")


# --------------------------------------------------------------
# 4. Prepare documents for BM25
# --------------------------------------------------------------

tokenized_corpus = []

for doc in corpus:

    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""

    combined_text = f"{title} {text}"

    tokens = tokenize(combined_text)

    tokenized_corpus.append(tokens)


# --------------------------------------------------------------
# 5. Build BM25 index
# --------------------------------------------------------------

print("Building BM25 index...")

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index ready.")


# --------------------------------------------------------------
# 6. Search function
# --------------------------------------------------------------

def search_bm25(query: str, k: int = 10):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    # Get indexes sorted from highest score to lowest
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []

    for rank, idx in enumerate(ranked_indexes, start=1):

        doc = corpus[idx]

        results.append({
            "rank": rank,
            "corpus_id": doc["corpus_id"],
            "score": float(scores[idx]),
            "title": doc.get("title", ""),
            "text": doc.get("text", "")
        })

    return results


# --------------------------------------------------------------
# 7. Interactive search
# --------------------------------------------------------------

if __name__ == "__main__":

    while True:

        print("\n" + "=" * 100)

        query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        if not query:
            continue

        results = search_bm25(
            query=query,
            k=TOP_K
        )

        print("\n" + "=" * 100)
        print(f"TOP {TOP_K} BM25 RESULTS")
        print("=" * 100)

        for result in results:

            print(f"\nRank:      {result['rank']}")
            print(f"Corpus ID: {result['corpus_id']}")
            print(f"BM25 Score:{result['score']:.4f}")

            if result["title"]:
                print(f"Title:     {result['title']}")

            text_preview = result["text"][:500].replace("\n", " ")

            print(f"Text:      {text_preview}")

            print("-" * 100)
