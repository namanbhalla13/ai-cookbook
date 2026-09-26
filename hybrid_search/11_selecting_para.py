"""
Tune BM25 + Vector + RRF + MMR parameters.

IMPORTANT:
- Does NOT modify 07_vector_serach.py
- Does NOT use DeepEval
- Does NOT use an LLM judge
- Uses existing ChromaDB
- Embeds every unique query only ONCE
- Reuses cached query embeddings across experiments
- Saves each completed configuration immediately

Pipeline:

Query
  |
  +---- BM25
  |
  +---- Chroma Vector Search
  |
  v
 RRF
  |
  v
 MMR
  |
  v
Final Top 10

Tune:
1. Base candidate K
2. RRF K
3. MMR lambda

Metrics:
- Recall@10
- Precision@10
- Hit@10
- MRR@10
- nDCG@10
"""

from pathlib import Path
import json
import math
import importlib.util

import numpy as np
import pandas as pd
import chromadb

from openai import OpenAI
from dotenv import load_dotenv


# ==============================================================
# ENVIRONMENT
# ==============================================================

load_dotenv()


# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "fiqa"

GOLD_PATH = (
    DATA_DIR
    / "gold_dataset_dev_100.json"
)

CHROMA_PATH = (
    DATA_DIR
    / "chroma_db"
)

OUTPUT_PATH = (
    DATA_DIR
    / "retriever_tuning_results.csv"
)


# ==============================================================
# CONFIG
# ==============================================================

COLLECTION_NAME = "fiqa_corpus"

EMBEDDING_MODEL = (
    "text-embedding-3-small"
)

TOP_K = 10


# ==============================================================
# PARAMETERS TO TEST
# ==============================================================

CANDIDATE_K_VALUES = [
    10,
    20,
    30,
    50,
]


RRF_K_VALUES = [
    20,
    40,
    60,
    80,
    100,
]


MMR_LAMBDA_VALUES = [
    0.6,
    0.7,
    0.8,
    0.9,
]


# ==============================================================
# IMPORT BM25
# ==============================================================

def load_module(
    file_name,
    module_name,
):

    file_path = (
        BASE_DIR
        / file_name
    )


    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            file_path,
        )
    )


    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"Could not import: "
            f"{file_path}"
        )


    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )


    spec.loader.exec_module(
        module
    )


    return module


bm25_module = load_module(
    "05_BM25.py",
    "bm25_module",
)


search_bm25 = (
    bm25_module.search_bm25
)


# ==============================================================
# OPENAI CLIENT
# ==============================================================

openai_client = OpenAI()


# ==============================================================
# CHROMA
# ==============================================================

chroma_client = (
    chromadb.PersistentClient(
        path=str(
            CHROMA_PATH
        )
    )
)


collection = (
    chroma_client.get_collection(
        name=COLLECTION_NAME
    )
)


print(
    f"Chroma documents: "
    f"{collection.count()}"
)


# ==============================================================
# QUERY EMBEDDING CACHE
# ==============================================================

query_embedding_cache = {}


def get_query_embedding(
    query,
):
    """
    Embed each unique query once.

    First request:
        OpenAI embedding call

    Every later use:
        reuse cached vector
    """

    query = str(
        query
    ).strip()


    if not query:

        raise ValueError(
            "Query cannot be empty."
        )


    # ----------------------------------------------------------
    # CACHE HIT
    # ----------------------------------------------------------

    if (
        query
        in query_embedding_cache
    ):

        return (
            query_embedding_cache[
                query
            ]
        )


    # ----------------------------------------------------------
    # CACHE MISS
    # ----------------------------------------------------------

    response = (
        openai_client
        .embeddings
        .create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
    )


    embedding = (
        response
        .data[0]
        .embedding
    )


    query_embedding_cache[
        query
    ] = embedding


    return embedding


# ==============================================================
# VECTOR SEARCH
#
# IMPORTANT:
# We query Chroma directly.
#
# Therefore 07_vector_serach.py does NOT need to change.
# ==============================================================

def search_vector_cached(
    query,
    k=10,
):

    # ----------------------------------------------------------
    # Get cached embedding
    # ----------------------------------------------------------

    query_embedding = (
        get_query_embedding(
            query
        )
    )


    # ----------------------------------------------------------
    # Search Chroma
    # ----------------------------------------------------------

    response = (
        collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=k,

            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )
    )


    results = []


    ids = (
        response["ids"][0]
    )

    documents = (
        response["documents"][0]
    )

    metadatas = (
        response["metadatas"][0]
    )

    distances = (
        response["distances"][0]
    )


    # ----------------------------------------------------------
    # Convert result format
    # ----------------------------------------------------------

    for rank, (
        corpus_id,
        document,
        metadata,
        distance,
    ) in enumerate(

        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),

        start=1,
    ):

        similarity = (
            1.0
            - float(distance)
        )


        results.append(
            {
                "rank":
                    rank,

                "corpus_id":
                    str(corpus_id),

                "score":
                    similarity,

                "title":
                    (
                        metadata.get(
                            "title",
                            "",
                        )
                        if metadata
                        else ""
                    ),

                "text":
                    (
                        document
                        or ""
                    ),
            }
        )


    return results


# ==============================================================
# DOCUMENT EMBEDDINGS
#
# These already exist inside Chroma.
#
# NO OpenAI call should normally be required here.
# ==============================================================

def get_document_embeddings(
    docs,
):

    ids = [
        str(
            doc["corpus_id"]
        )
        for doc
        in docs
    ]


    response = (
        collection.get(
            ids=ids,
            include=[
                "embeddings"
            ],
        )
    )


    embedding_map = {}


    embeddings = (
        response.get(
            "embeddings"
        )
    )


    if embeddings is None:

        raise RuntimeError(
            "Chroma did not return "
            "document embeddings."
        )


    for (
        corpus_id,
        embedding,
    ) in zip(
        response["ids"],
        embeddings,
    ):

        embedding_map[
            str(corpus_id)
        ] = embedding


    # ----------------------------------------------------------
    # Make sure every requested doc exists
    # ----------------------------------------------------------

    missing_ids = [

        corpus_id

        for corpus_id
        in ids

        if corpus_id
        not in embedding_map
    ]


    if missing_ids:

        raise RuntimeError(
            "Some candidate documents "
            "are missing from Chroma: "
            f"{missing_ids[:10]}"
        )


    return embedding_map


# ==============================================================
# COSINE SIMILARITY
# ==============================================================

def cosine_similarity(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=np.float32,
    )


    b = np.asarray(
        b,
        dtype=np.float32,
    )


    denominator = (

        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )


    if denominator == 0:

        return 0.0


    return float(

        np.dot(a, b)
        /
        denominator
    )


# ==============================================================
# RRF
# ==============================================================

def search_rrf_custom(
    query,
    base_k,
    rrf_k,
    final_k,
):

    # ----------------------------------------------------------
    # BM25
    # ----------------------------------------------------------

    bm25_results = (
        search_bm25(
            query=query,
            k=base_k,
        )
    )


    # ----------------------------------------------------------
    # VECTOR
    # ----------------------------------------------------------

    vector_results = (
        search_vector_cached(
            query=query,
            k=base_k,
        )
    )


    fused = {}


    # ----------------------------------------------------------
    # BM25 RANKS
    # ----------------------------------------------------------

    for rank, doc in enumerate(
        bm25_results,
        start=1,
    ):

        corpus_id = str(
            doc["corpus_id"]
        )


        if (
            corpus_id
            not in fused
        ):

            fused[
                corpus_id
            ] = {

                "corpus_id":
                    corpus_id,

                "title":
                    doc.get(
                        "title",
                        "",
                    ),

                "text":
                    doc.get(
                        "text",
                        "",
                    ),

                "rrf_score":
                    0.0,

                "bm25_rank":
                    None,

                "vector_rank":
                    None,
            }


        fused[
            corpus_id
        ][
            "bm25_rank"
        ] = rank


        fused[
            corpus_id
        ][
            "rrf_score"
        ] += (

            1.0
            /
            (
                rrf_k
                + rank
            )
        )


    # ----------------------------------------------------------
    # VECTOR RANKS
    # ----------------------------------------------------------

    for rank, doc in enumerate(
        vector_results,
        start=1,
    ):

        corpus_id = str(
            doc["corpus_id"]
        )


        if (
            corpus_id
            not in fused
        ):

            fused[
                corpus_id
            ] = {

                "corpus_id":
                    corpus_id,

                "title":
                    doc.get(
                        "title",
                        "",
                    ),

                "text":
                    doc.get(
                        "text",
                        "",
                    ),

                "rrf_score":
                    0.0,

                "bm25_rank":
                    None,

                "vector_rank":
                    None,
            }


        fused[
            corpus_id
        ][
            "vector_rank"
        ] = rank


        fused[
            corpus_id
        ][
            "rrf_score"
        ] += (

            1.0
            /
            (
                rrf_k
                + rank
            )
        )


    # ----------------------------------------------------------
    # SORT
    # ----------------------------------------------------------

    ranked = sorted(

        fused.values(),

        key=lambda x:
            x["rrf_score"],

        reverse=True,
    )


    return ranked[
        :final_k
    ]


# ==============================================================
# MMR
# ==============================================================

def search_mmr_custom(
    query,
    base_k,
    rrf_k,
    candidate_k,
    lambda_mult,
    final_k=10,
):

    # ----------------------------------------------------------
    # RRF CANDIDATES
    # ----------------------------------------------------------

    candidates = (
        search_rrf_custom(

            query=query,

            base_k=base_k,

            rrf_k=rrf_k,

            final_k=candidate_k,
        )
    )


    if not candidates:

        return []


    # ----------------------------------------------------------
    # SAME cached query embedding
    # ----------------------------------------------------------

    query_embedding = (
        get_query_embedding(
            query
        )
    )


    # ----------------------------------------------------------
    # Candidate embeddings already in Chroma
    # ----------------------------------------------------------

    embedding_map = (
        get_document_embeddings(
            candidates
        )
    )


    selected = []

    remaining = (
        candidates.copy()
    )


    # ==========================================================
    # MMR LOOP
    # ==============================================================

    while (
        remaining
        and len(selected)
        < final_k
    ):

        best_doc = None

        best_score = float(
            "-inf"
        )


        for doc in remaining:

            corpus_id = str(
                doc["corpus_id"]
            )


            doc_embedding = (
                embedding_map[
                    corpus_id
                ]
            )


            # --------------------------------------------------
            # RELEVANCE TO QUERY
            # --------------------------------------------------

            relevance = (
                cosine_similarity(
                    query_embedding,
                    doc_embedding,
                )
            )


            # --------------------------------------------------
            # REDUNDANCY
            # --------------------------------------------------

            if not selected:

                max_similarity = 0.0

            else:

                max_similarity = max(

                    cosine_similarity(

                        doc_embedding,

                        embedding_map[
                            str(
                                selected_doc[
                                    "corpus_id"
                                ]
                            )
                        ],
                    )

                    for selected_doc
                    in selected
                )


            # --------------------------------------------------
            # MMR
            # --------------------------------------------------

            mmr_score = (

                lambda_mult
                * relevance

                -

                (
                    1.0
                    - lambda_mult
                )
                * max_similarity
            )


            if (
                mmr_score
                > best_score
            ):

                best_score = (
                    mmr_score
                )


                best_doc = (
                    doc.copy()
                )


                best_doc[
                    "mmr_score"
                ] = mmr_score


                best_doc[
                    "query_similarity"
                ] = relevance


                best_doc[
                    "diversity_penalty"
                ] = (
                    max_similarity
                )


        if best_doc is None:

            break


        selected.append(
            best_doc
        )


        remaining = [

            doc

            for doc
            in remaining

            if str(
                doc[
                    "corpus_id"
                ]
            )

            !=

            str(
                best_doc[
                    "corpus_id"
                ]
            )
        ]


    return selected


# ==============================================================
# METRICS
# ==============================================================

def calculate_metrics(
    retrieved_ids,
    gold_ids,
    k=10,
):

    retrieved_ids = [

        str(x)

        for x
        in retrieved_ids[:k]
    ]


    gold_ids = {

        str(x)

        for x
        in gold_ids
    }


    # ----------------------------------------------------------
    # HITS
    # ----------------------------------------------------------

    hits = [

        doc_id

        for doc_id
        in retrieved_ids

        if doc_id
        in gold_ids
    ]


    unique_hits = set(
        hits
    )


    # ----------------------------------------------------------
    # RECALL
    # ----------------------------------------------------------

    recall = (

        len(unique_hits)
        / len(gold_ids)

        if gold_ids

        else 0.0
    )


    # ----------------------------------------------------------
    # PRECISION
    # ----------------------------------------------------------

    precision = (

        len(unique_hits)
        / len(retrieved_ids)

        if retrieved_ids

        else 0.0
    )


    # ----------------------------------------------------------
    # HIT
    # ----------------------------------------------------------

    hit = int(
        len(unique_hits) > 0
    )


    # ----------------------------------------------------------
    # MRR
    # ----------------------------------------------------------

    reciprocal_rank = 0.0


    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if doc_id in gold_ids:

            reciprocal_rank = (
                1.0 / rank
            )

            break


    # ----------------------------------------------------------
    # DCG
    # ----------------------------------------------------------

    dcg = 0.0


    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if doc_id in gold_ids:

            dcg += (

                1.0
                /
                math.log2(
                    rank + 1
                )
            )


    # ----------------------------------------------------------
    # IDEAL DCG
    # ----------------------------------------------------------

    ideal_count = min(
        len(gold_ids),
        k,
    )


    idcg = sum(

        1.0
        /
        math.log2(
            rank + 1
        )

        for rank
        in range(
            1,
            ideal_count + 1,
        )
    )


    ndcg = (

        dcg / idcg

        if idcg > 0

        else 0.0
    )


    return {

        "recall_at_10":
            recall,

        "precision_at_10":
            precision,

        "hit_at_10":
            hit,

        "mrr_at_10":
            reciprocal_rank,

        "ndcg_at_10":
            ndcg,
    }


# ==============================================================
# LOAD DEV DATA
# ==============================================================

with open(
    GOLD_PATH,
    "r",
    encoding="utf-8",
) as f:

    gold_data = (
        json.load(f)
    )


print(
    f"Loaded "
    f"{len(gold_data)} "
    f"DEV queries"
)


# ==============================================================
# EVALUATE CONFIGURATION
# ==============================================================

def evaluate_configuration(
    base_k,
    rrf_k,
    candidate_k,
    mmr_lambda,
):

    metric_rows = []


    for i, item in enumerate(
        gold_data,
        start=1,
    ):

        question = str(
            item["question"]
        )


        gold_ids = {

            str(
                doc[
                    "corpus_id"
                ]
            )

            for doc
            in item[
                "relevant_docs"
            ]
        }


        # ------------------------------------------------------
        # RETRIEVE
        # ------------------------------------------------------

        results = (
            search_mmr_custom(

                query=question,

                base_k=
                    base_k,

                rrf_k=
                    rrf_k,

                candidate_k=
                    candidate_k,

                lambda_mult=
                    mmr_lambda,

                final_k=
                    TOP_K,
            )
        )


        retrieved_ids = [

            str(
                doc[
                    "corpus_id"
                ]
            )

            for doc
            in results
        ]


        # ------------------------------------------------------
        # SCORE
        # ------------------------------------------------------

        metrics = (
            calculate_metrics(

                retrieved_ids=
                    retrieved_ids,

                gold_ids=
                    gold_ids,

                k=
                    TOP_K,
            )
        )


        metric_rows.append(
            metrics
        )


    df = pd.DataFrame(
        metric_rows
    )


    return {

        "base_k":
            base_k,

        "rrf_k":
            rrf_k,

        "candidate_k":
            candidate_k,

        "mmr_lambda":
            mmr_lambda,

        "recall_at_10":
            df[
                "recall_at_10"
            ].mean(),

        "precision_at_10":
            df[
                "precision_at_10"
            ].mean(),

        "hit_at_10":
            df[
                "hit_at_10"
            ].mean(),

        "mrr_at_10":
            df[
                "mrr_at_10"
            ].mean(),

        "ndcg_at_10":
            df[
                "ndcg_at_10"
            ].mean(),
    }


# ==============================================================
# PREVIOUS RESULTS
# ==============================================================

def load_previous_results():

    if not OUTPUT_PATH.exists():

        return pd.DataFrame()


    return pd.read_csv(
        OUTPUT_PATH
    )


# ==============================================================
# CHECK CONFIG
# ==============================================================

def config_exists(
    df,
    base_k,
    rrf_k,
    candidate_k,
    mmr_lambda,
):

    if df.empty:

        return False


    required = {
        "base_k",
        "rrf_k",
        "candidate_k",
        "mmr_lambda",
    }


    if not required.issubset(
        df.columns
    ):

        return False


    match = df[

        (
            df["base_k"]
            == base_k
        )

        &

        (
            df["rrf_k"]
            == rrf_k
        )

        &

        (
            df["candidate_k"]
            == candidate_k
        )

        &

        (
            df[
                "mmr_lambda"
            ].round(5)

            ==

            round(
                mmr_lambda,
                5,
            )
        )
    ]


    return not match.empty


# ==============================================================
# SAVE CONFIG RESULT
# ==============================================================

def save_result(
    result,
):

    result_df = pd.DataFrame(
        [result]
    )


    if not OUTPUT_PATH.exists():

        result_df.to_csv(
            OUTPUT_PATH,
            index=False,
        )


    else:

        result_df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=False,
            index=False,
        )


# ==============================================================
# RUN ONE CONFIG
# ==============================================================

def run_config(
    base_k,
    rrf_k,
    candidate_k,
    mmr_lambda,
):

    previous = (
        load_previous_results()
    )


    if config_exists(
        previous,
        base_k,
        rrf_k,
        candidate_k,
        mmr_lambda,
    ):

        print(
            f"SKIP | "
            f"base={base_k} | "
            f"rrf={rrf_k} | "
            f"candidate={candidate_k} | "
            f"lambda={mmr_lambda}"
        )

        return


    print(
        f"\nRUN | "
        f"base={base_k} | "
        f"rrf={rrf_k} | "
        f"candidate={candidate_k} | "
        f"lambda={mmr_lambda}"
    )


    result = (
        evaluate_configuration(

            base_k=
                base_k,

            rrf_k=
                rrf_k,

            candidate_k=
                candidate_k,

            mmr_lambda=
                mmr_lambda,
        )
    )


    save_result(
        result
    )


    print(
        f"Recall="
        f"{result['recall_at_10']:.4f} | "
        f"Precision="
        f"{result['precision_at_10']:.4f} | "
        f"Hit="
        f"{result['hit_at_10']:.4f} | "
        f"MRR="
        f"{result['mrr_at_10']:.4f} | "
        f"nDCG="
        f"{result['ndcg_at_10']:.4f}"
    )


# ==============================================================
# STAGE 1
# CANDIDATE SIZE
#
# Hold:
# RRF = 60
# lambda = 0.8
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 1 - CANDIDATE SIZE"
)

print(
    "=" * 80
)


for candidate_k in (
    CANDIDATE_K_VALUES
):

    run_config(

        base_k=
            candidate_k,

        rrf_k=
            60,

        candidate_k=
            candidate_k,

        mmr_lambda=
            0.8,
    )


# ==============================================================
# SELECT BEST CANDIDATE SIZE
# ==============================================================

results_df = (
    pd.read_csv(
        OUTPUT_PATH
    )
)


stage1 = results_df[

    (
        results_df[
            "rrf_k"
        ]
        == 60
    )

    &

    (
        results_df[
            "mmr_lambda"
        ].round(5)
        == 0.8
    )

    &

    (
        results_df[
            "base_k"
        ]
        == results_df[
            "candidate_k"
        ]
    )
]


stage1 = (
    stage1.sort_values(

        by=[
            "ndcg_at_10",
            "recall_at_10",
            "mrr_at_10",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )
)


best_candidate_k = int(
    stage1.iloc[0][
        "candidate_k"
    ]
)


print(
    f"\nBest candidate K: "
    f"{best_candidate_k}"
)


# ==============================================================
# STAGE 2
# RRF K
#
# Hold:
# candidate = best from Stage 1
# lambda = 0.8
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 2 - RRF K"
)

print(
    "=" * 80
)


for rrf_k in (
    RRF_K_VALUES
):

    run_config(

        base_k=
            best_candidate_k,

        rrf_k=
            rrf_k,

        candidate_k=
            best_candidate_k,

        mmr_lambda=
            0.8,
    )


# ==============================================================
# SELECT BEST RRF K
# ==============================================================

results_df = (
    pd.read_csv(
        OUTPUT_PATH
    )
)


stage2 = results_df[

    (
        results_df[
            "base_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "candidate_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "mmr_lambda"
        ].round(5)
        == 0.8
    )
]


stage2 = (
    stage2.sort_values(

        by=[
            "ndcg_at_10",
            "recall_at_10",
            "mrr_at_10",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )
)


best_rrf_k = int(
    stage2.iloc[0][
        "rrf_k"
    ]
)


print(
    f"\nBest RRF K: "
    f"{best_rrf_k}"
)


# ==============================================================
# STAGE 3
# MMR LAMBDA
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 3 - MMR LAMBDA"
)

print(
    "=" * 80
)


for mmr_lambda in (
    MMR_LAMBDA_VALUES
):

    run_config(

        base_k=
            best_candidate_k,

        rrf_k=
            best_rrf_k,

        candidate_k=
            best_candidate_k,

        mmr_lambda=
            mmr_lambda,
    )


# ==============================================================
# FINAL SELECTION
# ==============================================================

results_df = (
    pd.read_csv(
        OUTPUT_PATH
    )
)


final_candidates = results_df[

    (
        results_df[
            "base_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "candidate_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "rrf_k"
        ]
        == best_rrf_k
    )
]


final_candidates = (
    final_candidates.sort_values(

        by=[
            "ndcg_at_10",
            "recall_at_10",
            "mrr_at_10",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )
)


best = (
    final_candidates.iloc[0]
)


# ==============================================================
# FINAL OUTPUT
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "BEST CONFIGURATION"
)

print(
    "=" * 80
)


print(
    f"BM25 / Vector K : "
    f"{int(best['base_k'])}"
)

print(
    f"RRF K            : "
    f"{int(best['rrf_k'])}"
)

print(
    f"RRF → MMR pool   : "
    f"{int(best['candidate_k'])}"
)

print(
    f"MMR lambda       : "
    f"{best['mmr_lambda']:.2f}"
)


print(
    "\nMETRICS"
)


print(
    f"Recall@10    : "
    f"{best['recall_at_10']:.4f}"
)

print(
    f"Precision@10 : "
    f"{best['precision_at_10']:.4f}"
)

print(
    f"Hit@10       : "
    f"{best['hit_at_10']:.4f}"
)

print(
    f"MRR@10       : "
    f"{best['mrr_at_10']:.4f}"
)

print(
    f"nDCG@10      : "
    f"{best['ndcg_at_10']:.4f}"
)


print(
    "\n"
    + "=" * 80
)

print(
    "CACHE"
)

print(
    "=" * 80
)


print(
    f"Unique queries embedded: "
    f"{len(query_embedding_cache)}"
)


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)
"""
Tune BM25 + Vector + RRF + MMR parameters.

IMPORTANT:
- Does NOT modify 07_vector_serach.py
- Does NOT use DeepEval
- Does NOT use an LLM judge
- Uses existing ChromaDB
- Embeds every unique query only ONCE
- Reuses cached query embeddings across experiments
- Saves each completed configuration immediately

Pipeline:

Query
  |
  +---- BM25
  |
  +---- Chroma Vector Search
  |
  v
 RRF
  |
  v
 MMR
  |
  v
Final Top 10

Tune:
1. Base candidate K
2. RRF K
3. MMR lambda

Metrics:
- Recall@10
- Precision@10
- Hit@10
- MRR@10
- nDCG@10
"""

from pathlib import Path
import json
import math
import importlib.util

import numpy as np
import pandas as pd
import chromadb

from openai import OpenAI
from dotenv import load_dotenv


# ==============================================================
# ENVIRONMENT
# ==============================================================

load_dotenv()


# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "fiqa"

GOLD_PATH = (
    DATA_DIR
    / "gold_dataset_dev_100.json"
)

CHROMA_PATH = (
    DATA_DIR
    / "chroma_db"
)

OUTPUT_PATH = (
    DATA_DIR
    / "retriever_tuning_results.csv"
)


# ==============================================================
# CONFIG
# ==============================================================

COLLECTION_NAME = "fiqa_corpus"

EMBEDDING_MODEL = (
    "text-embedding-3-small"
)

TOP_K = 10


# ==============================================================
# PARAMETERS TO TEST
# ==============================================================

CANDIDATE_K_VALUES = [
    10,
    20,
    30,
    50,
]


RRF_K_VALUES = [
    20,
    40,
    60,
    80,
    100,
]


MMR_LAMBDA_VALUES = [
    0.6,
    0.7,
    0.8,
    0.9,
]


# ==============================================================
# IMPORT BM25
# ==============================================================

def load_module(
    file_name,
    module_name,
):

    file_path = (
        BASE_DIR
        / file_name
    )


    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            file_path,
        )
    )


    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"Could not import: "
            f"{file_path}"
        )


    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )


    spec.loader.exec_module(
        module
    )


    return module


bm25_module = load_module(
    "05_BM25.py",
    "bm25_module",
)


search_bm25 = (
    bm25_module.search_bm25
)


# ==============================================================
# OPENAI CLIENT
# ==============================================================

openai_client = OpenAI()


# ==============================================================
# CHROMA
# ==============================================================

chroma_client = (
    chromadb.PersistentClient(
        path=str(
            CHROMA_PATH
        )
    )
)


collection = (
    chroma_client.get_collection(
        name=COLLECTION_NAME
    )
)


print(
    f"Chroma documents: "
    f"{collection.count()}"
)


# ==============================================================
# QUERY EMBEDDING CACHE
# ==============================================================

query_embedding_cache = {}


def get_query_embedding(
    query,
):
    """
    Embed each unique query once.

    First request:
        OpenAI embedding call

    Every later use:
        reuse cached vector
    """

    query = str(
        query
    ).strip()


    if not query:

        raise ValueError(
            "Query cannot be empty."
        )


    # ----------------------------------------------------------
    # CACHE HIT
    # ----------------------------------------------------------

    if (
        query
        in query_embedding_cache
    ):

        return (
            query_embedding_cache[
                query
            ]
        )


    # ----------------------------------------------------------
    # CACHE MISS
    # ----------------------------------------------------------

    response = (
        openai_client
        .embeddings
        .create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
    )


    embedding = (
        response
        .data[0]
        .embedding
    )


    query_embedding_cache[
        query
    ] = embedding


    return embedding


# ==============================================================
# VECTOR SEARCH
#
# IMPORTANT:
# We query Chroma directly.
#
# Therefore 07_vector_serach.py does NOT need to change.
# ==============================================================

def search_vector_cached(
    query,
    k=10,
):

    # ----------------------------------------------------------
    # Get cached embedding
    # ----------------------------------------------------------

    query_embedding = (
        get_query_embedding(
            query
        )
    )


    # ----------------------------------------------------------
    # Search Chroma
    # ----------------------------------------------------------

    response = (
        collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=k,

            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )
    )


    results = []


    ids = (
        response["ids"][0]
    )

    documents = (
        response["documents"][0]
    )

    metadatas = (
        response["metadatas"][0]
    )

    distances = (
        response["distances"][0]
    )


    # ----------------------------------------------------------
    # Convert result format
    # ----------------------------------------------------------

    for rank, (
        corpus_id,
        document,
        metadata,
        distance,
    ) in enumerate(

        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),

        start=1,
    ):

        similarity = (
            1.0
            - float(distance)
        )


        results.append(
            {
                "rank":
                    rank,

                "corpus_id":
                    str(corpus_id),

                "score":
                    similarity,

                "title":
                    (
                        metadata.get(
                            "title",
                            "",
                        )
                        if metadata
                        else ""
                    ),

                "text":
                    (
                        document
                        or ""
                    ),
            }
        )


    return results


# ==============================================================
# DOCUMENT EMBEDDINGS
#
# These already exist inside Chroma.
#
# NO OpenAI call should normally be required here.
# ==============================================================

def get_document_embeddings(
    docs,
):

    ids = [
        str(
            doc["corpus_id"]
        )
        for doc
        in docs
    ]


    response = (
        collection.get(
            ids=ids,
            include=[
                "embeddings"
            ],
        )
    )


    embedding_map = {}


    embeddings = (
        response.get(
            "embeddings"
        )
    )


    if embeddings is None:

        raise RuntimeError(
            "Chroma did not return "
            "document embeddings."
        )


    for (
        corpus_id,
        embedding,
    ) in zip(
        response["ids"],
        embeddings,
    ):

        embedding_map[
            str(corpus_id)
        ] = embedding


    # ----------------------------------------------------------
    # Make sure every requested doc exists
    # ----------------------------------------------------------

    missing_ids = [

        corpus_id

        for corpus_id
        in ids

        if corpus_id
        not in embedding_map
    ]


    if missing_ids:

        raise RuntimeError(
            "Some candidate documents "
            "are missing from Chroma: "
            f"{missing_ids[:10]}"
        )


    return embedding_map


# ==============================================================
# COSINE SIMILARITY
# ==============================================================

def cosine_similarity(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=np.float32,
    )


    b = np.asarray(
        b,
        dtype=np.float32,
    )


    denominator = (

        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )


    if denominator == 0:

        return 0.0


    return float(

        np.dot(a, b)
        /
        denominator
    )


# ==============================================================
# RRF
# ==============================================================

def search_rrf_custom(
    query,
    base_k,
    rrf_k,
    final_k,
):

    # ----------------------------------------------------------
    # BM25
    # ----------------------------------------------------------

    bm25_results = (
        search_bm25(
            query=query,
            k=base_k,
        )
    )


    # ----------------------------------------------------------
    # VECTOR
    # ----------------------------------------------------------

    vector_results = (
        search_vector_cached(
            query=query,
            k=base_k,
        )
    )


    fused = {}


    # ----------------------------------------------------------
    # BM25 RANKS
    # ----------------------------------------------------------

    for rank, doc in enumerate(
        bm25_results,
        start=1,
    ):

        corpus_id = str(
            doc["corpus_id"]
        )


        if (
            corpus_id
            not in fused
        ):

            fused[
                corpus_id
            ] = {

                "corpus_id":
                    corpus_id,

                "title":
                    doc.get(
                        "title",
                        "",
                    ),

                "text":
                    doc.get(
                        "text",
                        "",
                    ),

                "rrf_score":
                    0.0,

                "bm25_rank":
                    None,

                "vector_rank":
                    None,
            }


        fused[
            corpus_id
        ][
            "bm25_rank"
        ] = rank


        fused[
            corpus_id
        ][
            "rrf_score"
        ] += (

            1.0
            /
            (
                rrf_k
                + rank
            )
        )


    # ----------------------------------------------------------
    # VECTOR RANKS
    # ----------------------------------------------------------

    for rank, doc in enumerate(
        vector_results,
        start=1,
    ):

        corpus_id = str(
            doc["corpus_id"]
        )


        if (
            corpus_id
            not in fused
        ):

            fused[
                corpus_id
            ] = {

                "corpus_id":
                    corpus_id,

                "title":
                    doc.get(
                        "title",
                        "",
                    ),

                "text":
                    doc.get(
                        "text",
                        "",
                    ),

                "rrf_score":
                    0.0,

                "bm25_rank":
                    None,

                "vector_rank":
                    None,
            }


        fused[
            corpus_id
        ][
            "vector_rank"
        ] = rank


        fused[
            corpus_id
        ][
            "rrf_score"
        ] += (

            1.0
            /
            (
                rrf_k
                + rank
            )
        )


    # ----------------------------------------------------------
    # SORT
    # ----------------------------------------------------------

    ranked = sorted(

        fused.values(),

        key=lambda x:
            x["rrf_score"],

        reverse=True,
    )


    return ranked[
        :final_k
    ]


# ==============================================================
# MMR
# ==============================================================

def search_mmr_custom(
    query,
    base_k,
    rrf_k,
    candidate_k,
    lambda_mult,
    final_k=10,
):

    # ----------------------------------------------------------
    # RRF CANDIDATES
    # ----------------------------------------------------------

    candidates = (
        search_rrf_custom(

            query=query,

            base_k=base_k,

            rrf_k=rrf_k,

            final_k=candidate_k,
        )
    )


    if not candidates:

        return []


    # ----------------------------------------------------------
    # SAME cached query embedding
    # ----------------------------------------------------------

    query_embedding = (
        get_query_embedding(
            query
        )
    )


    # ----------------------------------------------------------
    # Candidate embeddings already in Chroma
    # ----------------------------------------------------------

    embedding_map = (
        get_document_embeddings(
            candidates
        )
    )


    selected = []

    remaining = (
        candidates.copy()
    )


    # ==========================================================
    # MMR LOOP
    # ==============================================================

    while (
        remaining
        and len(selected)
        < final_k
    ):

        best_doc = None

        best_score = float(
            "-inf"
        )


        for doc in remaining:

            corpus_id = str(
                doc["corpus_id"]
            )


            doc_embedding = (
                embedding_map[
                    corpus_id
                ]
            )


            # --------------------------------------------------
            # RELEVANCE TO QUERY
            # --------------------------------------------------

            relevance = (
                cosine_similarity(
                    query_embedding,
                    doc_embedding,
                )
            )


            # --------------------------------------------------
            # REDUNDANCY
            # --------------------------------------------------

            if not selected:

                max_similarity = 0.0

            else:

                max_similarity = max(

                    cosine_similarity(

                        doc_embedding,

                        embedding_map[
                            str(
                                selected_doc[
                                    "corpus_id"
                                ]
                            )
                        ],
                    )

                    for selected_doc
                    in selected
                )


            # --------------------------------------------------
            # MMR
            # --------------------------------------------------

            mmr_score = (

                lambda_mult
                * relevance

                -

                (
                    1.0
                    - lambda_mult
                )
                * max_similarity
            )


            if (
                mmr_score
                > best_score
            ):

                best_score = (
                    mmr_score
                )


                best_doc = (
                    doc.copy()
                )


                best_doc[
                    "mmr_score"
                ] = mmr_score


                best_doc[
                    "query_similarity"
                ] = relevance


                best_doc[
                    "diversity_penalty"
                ] = (
                    max_similarity
                )


        if best_doc is None:

            break


        selected.append(
            best_doc
        )


        remaining = [

            doc

            for doc
            in remaining

            if str(
                doc[
                    "corpus_id"
                ]
            )

            !=

            str(
                best_doc[
                    "corpus_id"
                ]
            )
        ]


    return selected


# ==============================================================
# METRICS
# ==============================================================

def calculate_metrics(
    retrieved_ids,
    gold_ids,
    k=10,
):

    retrieved_ids = [

        str(x)

        for x
        in retrieved_ids[:k]
    ]


    gold_ids = {

        str(x)

        for x
        in gold_ids
    }


    # ----------------------------------------------------------
    # HITS
    # ----------------------------------------------------------

    hits = [

        doc_id

        for doc_id
        in retrieved_ids

        if doc_id
        in gold_ids
    ]


    unique_hits = set(
        hits
    )


    # ----------------------------------------------------------
    # RECALL
    # ----------------------------------------------------------

    recall = (

        len(unique_hits)
        / len(gold_ids)

        if gold_ids

        else 0.0
    )


    # ----------------------------------------------------------
    # PRECISION
    # ----------------------------------------------------------

    precision = (

        len(unique_hits)
        / len(retrieved_ids)

        if retrieved_ids

        else 0.0
    )


    # ----------------------------------------------------------
    # HIT
    # ----------------------------------------------------------

    hit = int(
        len(unique_hits) > 0
    )


    # ----------------------------------------------------------
    # MRR
    # ----------------------------------------------------------

    reciprocal_rank = 0.0


    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if doc_id in gold_ids:

            reciprocal_rank = (
                1.0 / rank
            )

            break


    # ----------------------------------------------------------
    # DCG
    # ----------------------------------------------------------

    dcg = 0.0


    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if doc_id in gold_ids:

            dcg += (

                1.0
                /
                math.log2(
                    rank + 1
                )
            )


    # ----------------------------------------------------------
    # IDEAL DCG
    # ----------------------------------------------------------

    ideal_count = min(
        len(gold_ids),
        k,
    )


    idcg = sum(

        1.0
        /
        math.log2(
            rank + 1
        )

        for rank
        in range(
            1,
            ideal_count + 1,
        )
    )


    ndcg = (

        dcg / idcg

        if idcg > 0

        else 0.0
    )


    return {

        "recall_at_10":
            recall,

        "precision_at_10":
            precision,

        "hit_at_10":
            hit,

        "mrr_at_10":
            reciprocal_rank,

        "ndcg_at_10":
            ndcg,
    }


# ==============================================================
# LOAD DEV DATA
# ==============================================================

with open(
    GOLD_PATH,
    "r",
    encoding="utf-8",
) as f:

    gold_data = (
        json.load(f)
    )


print(
    f"Loaded "
    f"{len(gold_data)} "
    f"DEV queries"
)


# ==============================================================
# EVALUATE CONFIGURATION
# ==============================================================

def evaluate_configuration(
    base_k,
    rrf_k,
    candidate_k,
    mmr_lambda,
):

    metric_rows = []


    for i, item in enumerate(
        gold_data,
        start=1,
    ):

        question = str(
            item["question"]
        )


        gold_ids = {

            str(
                doc[
                    "corpus_id"
                ]
            )

            for doc
            in item[
                "relevant_docs"
            ]
        }


        # ------------------------------------------------------
        # RETRIEVE
        # ------------------------------------------------------

        results = (
            search_mmr_custom(

                query=question,

                base_k=
                    base_k,

                rrf_k=
                    rrf_k,

                candidate_k=
                    candidate_k,

                lambda_mult=
                    mmr_lambda,

                final_k=
                    TOP_K,
            )
        )


        retrieved_ids = [

            str(
                doc[
                    "corpus_id"
                ]
            )

            for doc
            in results
        ]


        # ------------------------------------------------------
        # SCORE
        # ------------------------------------------------------

        metrics = (
            calculate_metrics(

                retrieved_ids=
                    retrieved_ids,

                gold_ids=
                    gold_ids,

                k=
                    TOP_K,
            )
        )


        metric_rows.append(
            metrics
        )


    df = pd.DataFrame(
        metric_rows
    )


    return {

        "base_k":
            base_k,

        "rrf_k":
            rrf_k,

        "candidate_k":
            candidate_k,

        "mmr_lambda":
            mmr_lambda,

        "recall_at_10":
            df[
                "recall_at_10"
            ].mean(),

        "precision_at_10":
            df[
                "precision_at_10"
            ].mean(),

        "hit_at_10":
            df[
                "hit_at_10"
            ].mean(),

        "mrr_at_10":
            df[
                "mrr_at_10"
            ].mean(),

        "ndcg_at_10":
            df[
                "ndcg_at_10"
            ].mean(),
    }


# ==============================================================
# PREVIOUS RESULTS
# ==============================================================

def load_previous_results():

    if not OUTPUT_PATH.exists():

        return pd.DataFrame()


    return pd.read_csv(
        OUTPUT_PATH
    )


# ==============================================================
# CHECK CONFIG
# ==============================================================

def config_exists(
    df,
    base_k,
    rrf_k,
    candidate_k,
    mmr_lambda,
):

    if df.empty:

        return False


    required = {
        "base_k",
        "rrf_k",
        "candidate_k",
        "mmr_lambda",
    }


    if not required.issubset(
        df.columns
    ):

        return False


    match = df[

        (
            df["base_k"]
            == base_k
        )

        &

        (
            df["rrf_k"]
            == rrf_k
        )

        &

        (
            df["candidate_k"]
            == candidate_k
        )

        &

        (
            df[
                "mmr_lambda"
            ].round(5)

            ==

            round(
                mmr_lambda,
                5,
            )
        )
    ]


    return not match.empty


# ==============================================================
# SAVE CONFIG RESULT
# ==============================================================

def save_result(
    result,
):

    result_df = pd.DataFrame(
        [result]
    )


    if not OUTPUT_PATH.exists():

        result_df.to_csv(
            OUTPUT_PATH,
            index=False,
        )


    else:

        result_df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=False,
            index=False,
        )


# ==============================================================
# RUN ONE CONFIG
# ==============================================================

def run_config(
    base_k,
    rrf_k,
    candidate_k,
    mmr_lambda,
):

    previous = (
        load_previous_results()
    )


    if config_exists(
        previous,
        base_k,
        rrf_k,
        candidate_k,
        mmr_lambda,
    ):

        print(
            f"SKIP | "
            f"base={base_k} | "
            f"rrf={rrf_k} | "
            f"candidate={candidate_k} | "
            f"lambda={mmr_lambda}"
        )

        return


    print(
        f"\nRUN | "
        f"base={base_k} | "
        f"rrf={rrf_k} | "
        f"candidate={candidate_k} | "
        f"lambda={mmr_lambda}"
    )


    result = (
        evaluate_configuration(

            base_k=
                base_k,

            rrf_k=
                rrf_k,

            candidate_k=
                candidate_k,

            mmr_lambda=
                mmr_lambda,
        )
    )


    save_result(
        result
    )


    print(
        f"Recall="
        f"{result['recall_at_10']:.4f} | "
        f"Precision="
        f"{result['precision_at_10']:.4f} | "
        f"Hit="
        f"{result['hit_at_10']:.4f} | "
        f"MRR="
        f"{result['mrr_at_10']:.4f} | "
        f"nDCG="
        f"{result['ndcg_at_10']:.4f}"
    )


# ==============================================================
# STAGE 1
# CANDIDATE SIZE
#
# Hold:
# RRF = 60
# lambda = 0.8
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 1 - CANDIDATE SIZE"
)

print(
    "=" * 80
)


for candidate_k in (
    CANDIDATE_K_VALUES
):

    run_config(

        base_k=
            candidate_k,

        rrf_k=
            60,

        candidate_k=
            candidate_k,

        mmr_lambda=
            0.8,
    )


# ==============================================================
# SELECT BEST CANDIDATE SIZE
# ==============================================================

results_df = (
    pd.read_csv(
        OUTPUT_PATH
    )
)


stage1 = results_df[

    (
        results_df[
            "rrf_k"
        ]
        == 60
    )

    &

    (
        results_df[
            "mmr_lambda"
        ].round(5)
        == 0.8
    )

    &

    (
        results_df[
            "base_k"
        ]
        == results_df[
            "candidate_k"
        ]
    )
]


stage1 = (
    stage1.sort_values(

        by=[
            "ndcg_at_10",
            "recall_at_10",
            "mrr_at_10",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )
)


best_candidate_k = int(
    stage1.iloc[0][
        "candidate_k"
    ]
)


print(
    f"\nBest candidate K: "
    f"{best_candidate_k}"
)


# ==============================================================
# STAGE 2
# RRF K
#
# Hold:
# candidate = best from Stage 1
# lambda = 0.8
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 2 - RRF K"
)

print(
    "=" * 80
)


for rrf_k in (
    RRF_K_VALUES
):

    run_config(

        base_k=
            best_candidate_k,

        rrf_k=
            rrf_k,

        candidate_k=
            best_candidate_k,

        mmr_lambda=
            0.8,
    )


# ==============================================================
# SELECT BEST RRF K
# ==============================================================

results_df = (
    pd.read_csv(
        OUTPUT_PATH
    )
)


stage2 = results_df[

    (
        results_df[
            "base_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "candidate_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "mmr_lambda"
        ].round(5)
        == 0.8
    )
]


stage2 = (
    stage2.sort_values(

        by=[
            "ndcg_at_10",
            "recall_at_10",
            "mrr_at_10",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )
)


best_rrf_k = int(
    stage2.iloc[0][
        "rrf_k"
    ]
)


print(
    f"\nBest RRF K: "
    f"{best_rrf_k}"
)


# ==============================================================
# STAGE 3
# MMR LAMBDA
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STAGE 3 - MMR LAMBDA"
)

print(
    "=" * 80
)


for mmr_lambda in (
    MMR_LAMBDA_VALUES
):

    run_config(

        base_k=
            best_candidate_k,

        rrf_k=
            best_rrf_k,

        candidate_k=
            best_candidate_k,

        mmr_lambda=
            mmr_lambda,
    )


# ==============================================================
# FINAL SELECTION
# ==============================================================

results_df = (
    pd.read_csv(
        OUTPUT_PATH
    )
)


final_candidates = results_df[

    (
        results_df[
            "base_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "candidate_k"
        ]
        == best_candidate_k
    )

    &

    (
        results_df[
            "rrf_k"
        ]
        == best_rrf_k
    )
]


final_candidates = (
    final_candidates.sort_values(

        by=[
            "ndcg_at_10",
            "recall_at_10",
            "mrr_at_10",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )
)


best = (
    final_candidates.iloc[0]
)


# ==============================================================
# FINAL OUTPUT
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "BEST CONFIGURATION"
)

print(
    "=" * 80
)


print(
    f"BM25 / Vector K : "
    f"{int(best['base_k'])}"
)

print(
    f"RRF K            : "
    f"{int(best['rrf_k'])}"
)

print(
    f"RRF → MMR pool   : "
    f"{int(best['candidate_k'])}"
)

print(
    f"MMR lambda       : "
    f"{best['mmr_lambda']:.2f}"
)


print(
    "\nMETRICS"
)


print(
    f"Recall@10    : "
    f"{best['recall_at_10']:.4f}"
)

print(
    f"Precision@10 : "
    f"{best['precision_at_10']:.4f}"
)

print(
    f"Hit@10       : "
    f"{best['hit_at_10']:.4f}"
)

print(
    f"MRR@10       : "
    f"{best['mrr_at_10']:.4f}"
)

print(
    f"nDCG@10      : "
    f"{best['ndcg_at_10']:.4f}"
)


print(
    "\n"
    + "=" * 80
)

print(
    "CACHE"
)

print(
    "=" * 80
)


print(
    f"Unique queries embedded: "
    f"{len(query_embedding_cache)}"
)


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)
