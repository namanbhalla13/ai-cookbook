"""
Learning-to-Rank reranker.

Pipeline:

BM25
  +
Vector
  ↓
RRF
  ↓
MMR candidate pool
  ↓
LTR
  ↓
FINAL top 10

LTR is trained using FiQA qrels while excluding
queries contained in gold_dataset_20.json.
"""

from pathlib import Path
import importlib.util
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "fiqa"

GOLD_PATH = (
    DATA_DIR / "gold_dataset_20.json"
)

QRELS_PATH = (
    DATA_DIR / "qrels.parquet"
)

QUERIES_PATH = (
    DATA_DIR / "queries.parquet"
)

MODEL_PATH = (
    DATA_DIR / "ltr_model.joblib"
)


# Limit training size initially
LTR_TRAIN_QUERIES = 200

FINAL_K = 10

LTR_CANDIDATE_K = 30


# ==============================================================
# IMPORT MODULES
# ==============================================================

def load_module(file_name, module_name):

    spec = importlib.util.spec_from_file_location(
        module_name,
        BASE_DIR / file_name,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


bm25_module = load_module(
    "05_BM25.py",
    "bm25_module"
)

vector_module = load_module(
    "07_vector_serach.py",
    "vector_module"
)

rrf_module = load_module(
    "08_rrf.py",
    "rrf_module"
)

mmr_module = load_module(
    "09_mmr.py",
    "mmr_module"
)


search_bm25 = bm25_module.search_bm25
search_vector = vector_module.search_vector
search_rrf = rrf_module.search_rrf
search_mmr = mmr_module.search_mmr


# ==============================================================
# FEATURE CREATION
# ==============================================================

def create_candidate_features(
    query,
    candidate_ids,
):

    bm25_results = search_bm25(
        query=query,
        k=10,
    )

    vector_results = search_vector(
        query=query,
        k=10,
    )

    rrf_results = search_rrf(
        query=query,
        k=20,
    )


    bm25_map = {
        str(doc["corpus_id"]): doc
        for doc in bm25_results
    }

    vector_map = {
        str(doc["corpus_id"]): doc
        for doc in vector_results
    }

    rrf_map = {
        str(doc["corpus_id"]): doc
        for doc in rrf_results
    }


    features = []


    for corpus_id in candidate_ids:

        corpus_id = str(
            corpus_id
        )


        bm25_doc = bm25_map.get(
            corpus_id
        )

        vector_doc = vector_map.get(
            corpus_id
        )

        rrf_doc = rrf_map.get(
            corpus_id
        )


        # Missing rank gets a bad rank
        bm25_rank = (
            bm25_doc["rank"]
            if bm25_doc
            else 100
        )

        vector_rank = (
            vector_doc["rank"]
            if vector_doc
            else 100
        )


        bm25_score = (
            float(bm25_doc["score"])
            if bm25_doc
            else 0.0
        )

        vector_score = (
            float(vector_doc["score"])
            if vector_doc
            else 0.0
        )

        rrf_score = (
            float(rrf_doc["rrf_score"])
            if rrf_doc
            else 0.0
        )


        features.append(
            [
                1 / bm25_rank,
                1 / vector_rank,
                bm25_score,
                vector_score,
                rrf_score,
            ]
        )


    return np.asarray(
        features,
        dtype=float,
    )


# ==============================================================
# TRAIN LTR
# ==============================================================

def train_ltr():

    print("Training LTR model...")


    # ----------------------------------------------------------
    # Evaluation query IDs
    # ----------------------------------------------------------

    with open(
        GOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        eval_data = json.load(f)


    eval_query_ids = {
        str(item["query_id"])
        for item in eval_data
    }


    # ----------------------------------------------------------
    # Load FiQA
    # ----------------------------------------------------------

    qrels = pd.read_parquet(
        QRELS_PATH
    )

    queries = pd.read_parquet(
        QUERIES_PATH
    )


    qrels["query-id"] = (
        qrels["query-id"]
        .astype(str)
    )

    qrels["corpus-id"] = (
        qrels["corpus-id"]
        .astype(str)
    )

    queries["_id"] = (
        queries["_id"]
        .astype(str)
    )


    # ----------------------------------------------------------
    # Remove evaluation queries
    # ----------------------------------------------------------

    train_qrels = qrels[
        ~qrels["query-id"].isin(
            eval_query_ids
        )
    ]


    train_query_ids = (
        train_qrels["query-id"]
        .unique()
        [:LTR_TRAIN_QUERIES]
    )


    X = []
    y = []


    # ==========================================================
    # BUILD TRAINING DATA
    # ==============================================================

    for i, query_id in enumerate(
        train_query_ids,
        start=1,
    ):

        query_row = queries[
            queries["_id"]
            == query_id
        ]

        if query_row.empty:
            continue


        question = (
            query_row.iloc[0]["text"]
        )


        gold_ids = set(
            train_qrels[
                train_qrels["query-id"]
                == query_id
            ]["corpus-id"]
        )


        # ------------------------------------------------------
        # Candidate pool from MMR
        # ------------------------------------------------------

        candidates = search_mmr(
            query=question,
            k=LTR_CANDIDATE_K,
            candidate_k=30,
        )


        candidate_ids = [
            str(doc["corpus_id"])
            for doc in candidates
        ]


        if not candidate_ids:
            continue


        feature_matrix = (
            create_candidate_features(
                query=question,
                candidate_ids=candidate_ids,
            )
        )


        labels = [
            1 if corpus_id in gold_ids
            else 0
            for corpus_id
            in candidate_ids
        ]


        X.extend(
            feature_matrix.tolist()
        )

        y.extend(
            labels
        )


        print(
            f"Training queries: "
            f"{i}/{len(train_query_ids)}"
        )


    X = np.asarray(X)
    y = np.asarray(y)


    print(
        f"Training rows: {len(X)}"
    )

    print(
        f"Positive rows: {y.sum()}"
    )


    if len(np.unique(y)) < 2:

        raise ValueError(
            "LTR training data does not contain "
            "both positive and negative examples."
        )


    # ----------------------------------------------------------
    # TRAIN MODEL
    # ----------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model.fit(
        X,
        y,
    )


    joblib.dump(
        model,
        MODEL_PATH,
    )


    print(
        f"LTR model saved to: "
        f"{MODEL_PATH}"
    )


    return model


# ==============================================================
# LOAD MODEL
# ==============================================================

def load_ltr_model():

    if MODEL_PATH.exists():

        return joblib.load(
            MODEL_PATH
        )

    raise FileNotFoundError(
        "\nLTR model does not exist.\n"
        "Run:\n"
        "python 10_ltr.py --train"
    )


# ==============================================================
# LTR SEARCH
# ==============================================================

def search_ltr(
    query: str,
    k: int = 10,
):

    model = load_ltr_model()


    # ----------------------------------------------------------
    # Take larger MMR candidate pool
    # ----------------------------------------------------------

    candidates = search_mmr(
        query=query,
        k=LTR_CANDIDATE_K,
        candidate_k=30,
    )


    candidate_ids = [
        str(doc["corpus_id"])
        for doc in candidates
    ]


    if not candidate_ids:
        return []


    # ----------------------------------------------------------
    # Features
    # ----------------------------------------------------------

    X = create_candidate_features(
        query=query,
        candidate_ids=candidate_ids,
    )


    # ----------------------------------------------------------
    # Probability of relevance
    # ----------------------------------------------------------

    relevance_scores = (
        model.predict_proba(X)[:, 1]
    )


    # ----------------------------------------------------------
    # Attach LTR scores
    # ----------------------------------------------------------

    reranked = []


    for doc, score in zip(
        candidates,
        relevance_scores,
    ):

        item = doc.copy()

        item["ltr_score"] = float(
            score
        )

        reranked.append(
            item
        )


    reranked.sort(
        key=lambda x: x["ltr_score"],
        reverse=True,
    )


    # ----------------------------------------------------------
    # FINAL TOP K
    # ----------------------------------------------------------

    results = []


    for rank, doc in enumerate(
        reranked[:k],
        start=1,
    ):

        doc["rank"] = rank
        doc["score"] = doc["ltr_score"]

        results.append(doc)


    return results


# ==============================================================
# CLI
# ==============================================================

if __name__ == "__main__":

    import sys


    if (
        len(sys.argv) > 1
        and sys.argv[1] == "--train"
    ):

        train_ltr()

    else:

        query = input(
            "Question: "
        )

        results = search_ltr(
            query=query,
            k=FINAL_K,
        )

        for doc in results:

            print(
                doc["rank"],
                doc["corpus_id"],
                round(
                    doc["ltr_score"],
                    4,
                ),
            )
