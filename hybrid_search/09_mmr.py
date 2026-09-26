"""
Maximum Marginal Relevance (MMR)

Pipeline:

BM25 + Vector
      ↓
     RRF
      ↓
up to 20 candidates
      ↓
     MMR
      ↓
top K diverse + relevant documents
"""

from pathlib import Path
import importlib.util
import numpy as np


BASE_DIR = Path(__file__).parent

MMR_LAMBDA = 0.7


def load_module(file_name, module_name):

    spec = importlib.util.spec_from_file_location(
        module_name,
        BASE_DIR / file_name,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ==============================================================
# IMPORT RRF
# ==============================================================

rrf_module = load_module(
    "08_rrf.py",
    "rrf_module"
)

search_rrf = rrf_module.search_rrf


# ==============================================================
# IMPORT VECTOR MODULE
# ==============================================================

vector_module = load_module(
    "07_vector_serach.py",
    "vector_module"
)

collection = vector_module.collection
create_embeddings = vector_module.create_embeddings


# ==============================================================
# COSINE SIMILARITY
# ==============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ==============================================================
# GET DOCUMENT EMBEDDINGS FROM CHROMA
# ==============================================================

def get_document_embeddings(docs):

    ids = [
        str(doc["corpus_id"])
        for doc in docs
    ]

    response = collection.get(
        ids=ids,
        include=["embeddings"],
    )

    embedding_map = {}

    if response["embeddings"] is not None:

        for corpus_id, embedding in zip(
            response["ids"],
            response["embeddings"],
        ):

            embedding_map[
                str(corpus_id)
            ] = embedding


    # ----------------------------------------------------------
    # Fallback if an embedding is missing
    # ----------------------------------------------------------

    for doc in docs:

        corpus_id = str(
            doc["corpus_id"]
        )

        if corpus_id not in embedding_map:

            embedding = create_embeddings(
                [doc["text"]]
            )[0]

            embedding_map[
                corpus_id
            ] = embedding


    return embedding_map


# ==============================================================
# MMR SEARCH
# ==============================================================

def search_mmr(
    query: str,
    k: int = 10,
    candidate_k: int = 20,
    lambda_mult: float = 0.8,
):

    # ----------------------------------------------------------
    # RRF candidates
    # ----------------------------------------------------------

    candidates = search_rrf(
        query=query,
        k=candidate_k,
        rrf_k=60,
        base_k=20,
    )

    if not candidates:
        return []


    # ----------------------------------------------------------
    # Query embedding
    # ----------------------------------------------------------

    query_embedding = create_embeddings(
        [query]
    )[0]


    # ----------------------------------------------------------
    # Candidate embeddings
    # ----------------------------------------------------------

    embedding_map = get_document_embeddings(
        candidates
    )


    selected = []
    remaining = candidates.copy()


    # ==========================================================
    # MMR LOOP
    # ==============================================================

    while remaining and len(selected) < k:

        best_doc = None
        best_score = float("-inf")


        for doc in remaining:

            corpus_id = str(
                doc["corpus_id"]
            )

            doc_embedding = embedding_map[
                corpus_id
            ]


            # --------------------------------------------------
            # Query relevance
            # --------------------------------------------------

            relevance = cosine_similarity(
                query_embedding,
                doc_embedding,
            )


            # --------------------------------------------------
            # Diversity penalty
            # --------------------------------------------------

            if not selected:

                max_similarity = 0.0

            else:

                similarities = []

                for selected_doc in selected:

                    selected_embedding = (
                        embedding_map[
                            str(
                                selected_doc[
                                    "corpus_id"
                                ]
                            )
                        ]
                    )

                    similarities.append(
                        cosine_similarity(
                            doc_embedding,
                            selected_embedding,
                        )
                    )

                max_similarity = max(
                    similarities
                )


            # --------------------------------------------------
            # MMR
            # --------------------------------------------------

            mmr_score = (
                lambda_mult * relevance
                -
                (1 - lambda_mult)
                * max_similarity
            )


            if mmr_score > best_score:

                best_score = mmr_score
                best_doc = doc.copy()

                best_doc[
                    "query_similarity"
                ] = relevance

                best_doc[
                    "diversity_penalty"
                ] = max_similarity

                best_doc[
                    "mmr_score"
                ] = mmr_score


        selected.append(
            best_doc
        )

        remaining = [
            doc
            for doc in remaining
            if str(doc["corpus_id"])
            != str(best_doc["corpus_id"])
        ]


    # ----------------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------------

    results = []

    for rank, doc in enumerate(
        selected,
        start=1,
    ):

        doc["rank"] = rank
        doc["score"] = doc["mmr_score"]

        results.append(doc)


    return results


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    query = input("Question: ")

    results = search_mmr(
        query=query,
        k=10,
    )

    for doc in results:

        print(
            doc["rank"],
            doc["corpus_id"],
            round(doc["mmr_score"], 4),
        )
