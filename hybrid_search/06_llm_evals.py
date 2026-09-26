"""
Incremental evaluation of retrieval strategies against FiQA.

Already evaluated:
- BM25
- VECTOR

Evaluate:
- RRF
- MMR
- LTR

Metrics:
- Exact Recall@10
- Exact Precision@10
- Hit@10
- DeepEval Contextual Recall
- DeepEval Contextual Precision

Important behavior:
- Saves EVERY completed query immediately to CSV.
- If script crashes, completed query evaluations are preserved.
- On restart, already completed query/retriever combinations are skipped.
- If the LTR model does not exist, LTR training starts automatically.
"""

from pathlib import Path
import json
import pandas as pd
import importlib.util
import subprocess
import sys
import math

from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models import OpenAIModel


# ==============================================================
# ENVIRONMENT
# ==============================================================

load_dotenv()


# ==============================================================
# MODULE LOADER
# ==============================================================

def load_module(file_name, module_name):

    file_path = Path(__file__).parent / file_name

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module from {file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ==============================================================
# IMPORT RETRIEVERS
# ==============================================================

rrf_module = load_module(
    "08_rrf.py",
    "rrf_module",
)

search_rrf = rrf_module.search_rrf


mmr_module = load_module(
    "09_mmr.py",
    "mmr_module",
)

search_mmr = mmr_module.search_mmr


ltr_module = load_module(
    "10_ltr.py",
    "ltr_module",
)

search_ltr = ltr_module.search_ltr


# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "fiqa"

GOLD_PATH = (
    DATA_DIR
    / "gold_dataset_20.json"
)

OUTPUT_PATH = (
    DATA_DIR
    / "retriever_comparison_results.csv"
)

SUMMARY_PATH = (
    DATA_DIR
    / "retriever_comparison_summary.csv"
)

LTR_MODEL_PATH = (
    DATA_DIR
    / "ltr_model.joblib"
)

LTR_SCRIPT_PATH = (
    BASE_DIR
    / "10_ltr.py"
)

TOP_K = 10


# ==============================================================
# LLM JUDGE
# ==============================================================

judge_model = OpenAIModel(
    model="gpt-5.6-luna",
    temperature=1,
)


# ==============================================================
# DEEPEVAL METRICS
# ==============================================================

recall_metric = ContextualRecallMetric(
    threshold=0.5,
    model=judge_model,
    include_reason=True,
    strict_mode=False,
)

precision_metric = ContextualPrecisionMetric(
    threshold=0.5,
    model=judge_model,
    include_reason=True,
    strict_mode=False,
)


# ==============================================================
# LOAD GOLD DATA
# ==============================================================

with open(
    GOLD_PATH,
    "r",
    encoding="utf-8",
) as f:

    gold_data = json.load(f)


print(
    f"Loaded {len(gold_data)} gold queries"
)


# ==============================================================
# LOAD EXISTING RESULTS
# ==============================================================

def load_existing_results():

    if not OUTPUT_PATH.exists():

        return pd.DataFrame()


    df = pd.read_csv(
        OUTPUT_PATH,
        dtype={
            "query_id": str
        },
    )


    if (
        not df.empty
        and "retriever" in df.columns
    ):

        print(
            "\nExisting saved results:"
        )

        print(
            df["retriever"]
            .value_counts()
            .to_string()
        )


    return df


# ==============================================================
# SAVE ONE QUERY RESULT
# ==============================================================

def save_result(row):
    """
    Save one completed query immediately.

    If the same retriever + query_id already exists,
    replace it rather than duplicating it.
    """

    new_row = pd.DataFrame(
        [row]
    )


    # ----------------------------------------------------------
    # First ever result
    # ----------------------------------------------------------

    if not OUTPUT_PATH.exists():

        new_row.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        return


    # ----------------------------------------------------------
    # Existing CSV
    # ----------------------------------------------------------

    existing_df = pd.read_csv(
        OUTPUT_PATH,
        dtype={
            "query_id": str
        },
    )


    if not existing_df.empty:

        duplicate_mask = (

            (
                existing_df["retriever"]
                == row["retriever"]
            )

            &

            (
                existing_df["query_id"]
                .astype(str)
                == str(row["query_id"])
            )
        )


        # Remove old copy if one exists
        existing_df = existing_df[
            ~duplicate_mask
        ]


    updated_df = pd.concat(
        [
            existing_df,
            new_row,
        ],
        ignore_index=True,
    )


    updated_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


# ==============================================================
# COMPLETED QUERY IDS
# ==============================================================

def get_completed_query_ids(
    retriever_name,
):

    existing_df = (
        load_existing_results()
    )


    if existing_df.empty:
        return set()


    if "retriever" not in existing_df.columns:
        return set()


    rows = existing_df[
        existing_df["retriever"]
        == retriever_name
    ]


    if rows.empty:
        return set()


    return set(
        rows["query_id"]
        .astype(str)
    )


# ==============================================================
# EVALUATE ONE RETRIEVER
# ==============================================================

def evaluate_retriever(
    retriever_name,
    search_function,
    gold_data,
):

    completed_query_ids = (
        get_completed_query_ids(
            retriever_name
        )
    )


    gold_query_ids = {
        str(item["query_id"])
        for item in gold_data
    }


    print(
        f"\n{retriever_name}: "
        f"{len(completed_query_ids)}/"
        f"{len(gold_query_ids)} "
        f"queries already saved."
    )


    # ----------------------------------------------------------
    # Retriever already completely evaluated
    # ----------------------------------------------------------

    if gold_query_ids.issubset(
        completed_query_ids
    ):

        print(
            f"{retriever_name} already complete. "
            f"Skipping evaluation."
        )

        return


    # ==========================================================
    # QUERY LOOP
    # ==============================================================

    for i, item in enumerate(
        gold_data,
        start=1,
    ):

        query_id = str(
            item["query_id"]
        )


        # ------------------------------------------------------
        # Already completed
        # ------------------------------------------------------

        if query_id in completed_query_ids:

            print(
                f"{retriever_name} | "
                f"{i}/{len(gold_data)} | "
                f"already saved"
            )

            continue


        question = item[
            "question"
        ]

        gold_docs = item[
            "relevant_docs"
        ]


        # ------------------------------------------------------
        # RETRIEVE
        # ------------------------------------------------------

        retrieved_docs = search_function(
            query=question,
            k=TOP_K,
        )


        # ------------------------------------------------------
        # RETRIEVED CONTEXTS
        # ------------------------------------------------------

        retrieved_contexts = [
            str(doc.get("text", "") or "")
            for doc in retrieved_docs
        ]


        # ------------------------------------------------------
        # GOLD REFERENCE
        # ------------------------------------------------------

        expected_output = "\n\n".join(
            str(
                doc.get("text", "")
                or ""
            )
            for doc in gold_docs
        )


        # ------------------------------------------------------
        # DEEPEVAL TEST CASE
        # ------------------------------------------------------

        test_case = LLMTestCase(
            input=question,
            actual_output=expected_output,
            expected_output=expected_output,
            retrieval_context=retrieved_contexts,
        )


        # ------------------------------------------------------
        # CONTEXT RECALL
        # ------------------------------------------------------

        recall_metric.measure(
            test_case
        )


        context_recall = float(
            recall_metric.score
        )

        recall_reason = (
            recall_metric.reason
        )


        # ------------------------------------------------------
        # CONTEXT PRECISION
        # ------------------------------------------------------

        precision_metric.measure(
            test_case
        )


        context_precision = float(
            precision_metric.score
        )

        precision_reason = (
            precision_metric.reason
        )


        # ------------------------------------------------------
        # GOLD IDS
        # ------------------------------------------------------

        gold_ids = {
            str(doc["corpus_id"])
            for doc in gold_docs
        }


        # ------------------------------------------------------
        # RETRIEVED IDS
        # ------------------------------------------------------

        retrieved_ids = [
            str(doc["corpus_id"])
            for doc in retrieved_docs
        ]


        # ------------------------------------------------------
        # HITS
        # ------------------------------------------------------

        hits = [
            corpus_id
            for corpus_id
            in retrieved_ids
            if corpus_id
            in gold_ids
        ]


        unique_hits = set(
            hits
        )

        # ------------------------------------------------------
        # MRR@K
        # ------------------------------------------------------

        reciprocal_rank = 0.0

        for rank, doc_id in enumerate(retrieved_ids, start=1):

            if doc_id in gold_ids:
                reciprocal_rank = 1.0 / rank
                break


        # ------------------------------------------------------
        # nDCG@K
        # ------------------------------------------------------

        dcg = 0.0

        for rank, doc_id in enumerate(retrieved_ids, start=1):

            relevance = 1 if doc_id in gold_ids else 0

            if relevance > 0:
                dcg += relevance / math.log2(rank + 1)


        ideal_relevant_count = min(
            len(gold_ids),
            TOP_K
        )

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(
                1,
                ideal_relevant_count + 1
            )
        )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0.0
        )

        # ------------------------------------------------------
        # EXACT RECALL@K
        # ------------------------------------------------------

        exact_recall = (

            len(unique_hits)
            / len(gold_ids)

            if gold_ids

            else 0.0
        )


        # ------------------------------------------------------
        # EXACT PRECISION@K
        # ------------------------------------------------------

        exact_precision = (

            len(unique_hits)
            / len(retrieved_ids)

            if retrieved_ids

            else 0.0
        )


        # ------------------------------------------------------
        # HIT@K
        # ------------------------------------------------------

        hit_at_k = int(
            len(unique_hits) > 0
        )


        # ------------------------------------------------------
        # RESULT ROW
        # ------------------------------------------------------

        row = {

            "retriever": retriever_name,

            "query_id": query_id,

            "question": question,

            "gold_ids": list(gold_ids),

            "retrieved_ids": retrieved_ids,

            "hits": hits,

            f"exact_recall_at_{TOP_K}":
                exact_recall,

            f"exact_precision_at_{TOP_K}":
                exact_precision,

            f"hit_at_{TOP_K}":
                hit_at_k,

            f"mrr_at_{TOP_K}":
                reciprocal_rank,

            f"ndcg_at_{TOP_K}":
                ndcg,

            "context_recall":
                context_recall,

            "context_precision":
                context_precision,

            "context_recall_reason":
                recall_reason,

            "context_precision_reason":
                precision_reason,
        }

        # ======================================================
        # SAVE IMMEDIATELY
        # ======================================================

        save_result(
            row
        )


        completed_query_ids.add(
            query_id
        )


        print(
            f"{retriever_name} | "
            f"{i}/{len(gold_data)} | "
            f"Recall@{TOP_K}: "
            f"{exact_recall:.2f} | "
            f"LLM Recall: "
            f"{context_recall:.2f} | "
            f"LLM Precision: "
            f"{context_precision:.2f} | "
            f"SAVED"
        )


    print(
        f"\n{retriever_name} complete."
    )


# ==============================================================
# RRF
# ==============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RRF EVALUATION"
)

print(
    "=" * 70
)


evaluate_retriever(
    retriever_name="RRF",
    search_function=search_rrf,
    gold_data=gold_data,
)


# ==============================================================
# MMR
# ==============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MMR EVALUATION"
)

print(
    "=" * 70
)


evaluate_retriever(
    retriever_name="MMR",
    search_function=search_mmr,
    gold_data=gold_data,
)


# ==============================================================
# LTR MODEL TRAINING
# ==============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "LTR"
)

print(
    "=" * 70
)


# --------------------------------------------------------------
# Does the model already exist?
# --------------------------------------------------------------

if not LTR_MODEL_PATH.exists():

    print(
        "LTR model not found."
    )

    print(
        "Starting automatic LTR training..."
    )

    print(
        f"Command:"
    )

    print(
        f"{sys.executable} "
        f"{LTR_SCRIPT_PATH} "
        f"--train"
    )


    # ----------------------------------------------------------
    # Run training in the SAME virtual environment
    # ----------------------------------------------------------

    subprocess.run(
        [
            sys.executable,
            str(LTR_SCRIPT_PATH),
            "--train",
        ],
        check=True,
        cwd=str(BASE_DIR),
    )


# --------------------------------------------------------------
# Verify model exists after training
# --------------------------------------------------------------

if not LTR_MODEL_PATH.exists():

    raise FileNotFoundError(
        "\nLTR training process finished, "
        "but the model file was not created:\n"
        f"{LTR_MODEL_PATH}"
    )


print(
    f"LTR model ready: "
    f"{LTR_MODEL_PATH}"
)


# ==============================================================
# LTR EVALUATION
# ==============================================================

evaluate_retriever(
    retriever_name="LTR",
    search_function=search_ltr,
    gold_data=gold_data,
)


# ==============================================================
# BUILD FINAL SUMMARY
# ==============================================================

if OUTPUT_PATH.exists():

    final_df = pd.read_csv(
        OUTPUT_PATH,
        dtype={
            "query_id": str
        },
    )


    summary = (
            final_df
            .groupby("retriever")
            .agg(
                {
                    f"exact_recall_at_{TOP_K}": "mean",
                    f"exact_precision_at_{TOP_K}": "mean",
                    f"hit_at_{TOP_K}": "mean",
                    f"mrr_at_{TOP_K}": "mean",
                    f"ndcg_at_{TOP_K}": "mean",
                    "context_recall": "mean",
                    "context_precision": "mean",
                }
            )
            .reset_index()
        )


    # ----------------------------------------------------------
    # ORDER
    # ----------------------------------------------------------

    retriever_order = [
        "BM25",
        "VECTOR",
        "RRF",
        "MMR",
        "LTR",
    ]


    summary[
        "retriever"
    ] = pd.Categorical(
        summary["retriever"],
        categories=retriever_order,
        ordered=True,
    )


    summary = (
        summary
        .sort_values(
            "retriever"
        )
        .reset_index(
            drop=True
        )
    )


    # ----------------------------------------------------------
    # SAVE SUMMARY
    # ----------------------------------------------------------

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )


    # ----------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL RETRIEVER COMPARISON"
    )

    print(
        "=" * 100
    )


    print(
        summary.to_string(
            index=False
        )
    )


    print(
        f"\nDetailed results:\n"
        f"{OUTPUT_PATH}"
    )


    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )
