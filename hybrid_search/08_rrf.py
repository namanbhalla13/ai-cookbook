"""
Reciprocal Rank Fusion (RRF)

Combines:
- BM25 top 10
- Vector top 10

Returns fused ranking.
"""

from pathlib import Path
import importlib.util


BASE_DIR = Path(__file__).parent

RRF_K = 60
BASE_RETRIEVER_K = 10


def load_module(file_name, module_name):

    spec = importlib.util.spec_from_file_location(
        module_name,
        BASE_DIR / file_name,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ==============================================================
# IMPORT RETRIEVERS
# ==============================================================

bm25_module = load_module(
    "05_BM25.py",
    "bm25_module"
)

vector_module = load_module(
    "07_vector_serach.py",
    "vector_module"
)

search_bm25 = bm25_module.search_bm25
search_vector = vector_module.search_vector


# ==============================================================
# RRF SEARCH
# ==============================================================

def search_rrf(
    query: str,
    k: int = 10,
    rrf_k: int = 60,
    base_k: int = 20,
):

    bm25_results = search_bm25(
        query=query,
        k=base_k,
    )

    vector_results = search_vector(
        query=query,
        k=base_k,
    )

    fused = {}


    # ----------------------------------------------------------
    # BM25
    # ----------------------------------------------------------

    for rank, doc in enumerate(
        bm25_results,
        start=1,
    ):

        corpus_id = str(doc["corpus_id"])

        if corpus_id not in fused:

            fused[corpus_id] = {
                "corpus_id": corpus_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "rrf_score": 0.0,

                "bm25_rank": None,
                "bm25_score": None,

                "vector_rank": None,
                "vector_score": None,
            }

        fused[corpus_id]["bm25_rank"] = rank
        fused[corpus_id]["bm25_score"] = doc["score"]

        fused[corpus_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )


    # ----------------------------------------------------------
    # VECTOR
    # ----------------------------------------------------------

    for rank, doc in enumerate(
        vector_results,
        start=1,
    ):

        corpus_id = str(doc["corpus_id"])

        if corpus_id not in fused:

            fused[corpus_id] = {
                "corpus_id": corpus_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "rrf_score": 0.0,

                "bm25_rank": None,
                "bm25_score": None,

                "vector_rank": None,
                "vector_score": None,
            }

        fused[corpus_id]["vector_rank"] = rank
        fused[corpus_id]["vector_score"] = doc["score"]

        fused[corpus_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )


    # ----------------------------------------------------------
    # SORT
    # ----------------------------------------------------------

    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )


    # ----------------------------------------------------------
    # FINAL RESULTS
    # ----------------------------------------------------------

    results = []

    for rank, doc in enumerate(
        ranked[:k],
        start=1,
    ):

        doc["rank"] = rank
        doc["score"] = doc["rrf_score"]

        results.append(doc)


    return results


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    query = input("Question: ")

    results = search_rrf(
        query=query,
        k=10,
    )

    for doc in results:

        print(
            doc["rank"],
            doc["corpus_id"],
            round(doc["rrf_score"], 6),
            "BM25:",
            doc["bm25_rank"],
            "Vector:",
            doc["vector_rank"],
        )
