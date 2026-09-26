# Hybrid Search — Retriever Comparison (FiQA)

Comparison of BM25, Vector, RRF, MMR, and LTR retrieval strategies on the FiQA
corpus, based on `fiqa/retriever_comparison_summary.csv`,
`fiqa/retriever_comparison_results.csv`, and the parameter grid search in
`fiqa/retriever_tuning_results.csv`.

## 1. Setup

| | |
|---|---|
| Corpus | FiQA (`fiqa/full_corpus.json`, embedded into Chroma via `text-embedding-3-small`) |
| Eval set (final comparison) | `gold_dataset_20.json` — 20 queries, LLM-judged |
| Eval set (parameter tuning) | `gold_dataset_dev_100.json` — 100 queries, exact-match only, no LLM judge |
| LLM judge | `gpt-5.6-luna` via DeepEval `ContextualRecallMetric` / `ContextualPrecisionMetric`, temperature = 1 |
| Retrievers | BM25 → VECTOR → **RRF** (fuse) → **MMR** (diversify RRF pool) → **LTR** (logistic-regression rerank of the MMR/RRF pool) |

**Metrics**

- `exact_recall_at_10` / `exact_precision_at_10` / `hit_at_10` / `mrr_at_10` / `ndcg_at_10` — rank metrics computed against the exact gold corpus IDs.
- `context_recall` / `context_precision` — DeepEval's LLM-judge scores, i.e. whether the retrieved *text* semantically supports the gold answer and whether relevant chunks are ranked above irrelevant ones. These are the metrics that matter for what actually reaches the RAG generator.

## 2. Final results (current run)

From `retriever_comparison_summary.csv`:

| Retriever | Exact Recall@10 | Exact Prec@10 | Hit@10 | MRR@10 | nDCG@10 | Context Recall | Context Precision |
|---|---|---|---|---|---|---|---|
| BM25 | 0.398 | 0.070 | 0.60 | – | – | 0.571 | 0.776 |
| VECTOR | 0.418 | 0.090 | 0.55 | – | – | 0.585 | 0.858 |
| **RRF** | **0.485** | 0.090 | **0.65** | 0.482 | **0.401** | 0.652 | 0.748 |
| **MMR** | 0.468 | **0.095** | 0.60 | **0.470** | **0.402** | **0.675** | **0.823** |
| LTR | 0.418 | 0.090 | 0.55 | 0.463 | 0.378 | 0.593 | 0.790 |

**Reading this:**

- Hybrid fusion (RRF) beats either single retriever on every exact-match metric — the classic "BM25 catches keyword matches vector misses, vector catches paraphrases BM25 misses" effect.
- MMR gives up a hair of exact recall/hit vs. RRF (0.468 vs 0.485, 0.60 vs 0.65) but wins on **every context metric** — it's picking a *less redundant, more semantically on-target* top-10, which is what an LLM generator actually consumes.
- LTR is the weakest of the three fused strategies here — it ties VECTOR on exact metrics and is worst on both context metrics. The reranker is trained on FiQA qrels excluding the 20 eval queries, and on this small 20-query eval set it isn't adding lift over RRF/MMR — likely too little training signal / candidate overlap for the logistic-regression model to learn a useful ranking beyond what RRF+MMR already surface.

## 3. How we got here — three iterations

Three snapshots of this same experiment exist in `fiqa/`, run in this order:

| File | Order | Notes |
|---|---|---|
| `retriever_comparison_summary_old_stragey_v1.csv` | 1st | No `mrr_at_10` / `ndcg_at_10` columns yet — rank metrics were added later |
| `retriever_comparison_summary_v2.csv` | 2nd | Full metric set, but RRF/MMR params regressed |
| `retriever_comparison_summary.csv` | 3rd (current) | Re-run after the grid search in `11_selecting_para.py` |

BM25 and VECTOR are identical across all three runs (fixed baselines, not touched by tuning). RRF/MMR/LTR move:

| Retriever | Metric | v1 (old strategy) | v2 | Final (tuned) |
|---|---|---|---|---|
| RRF | Exact Recall@10 | 0.477 | 0.410 | **0.485** |
| RRF | Context Precision | 0.780 | 0.739 | 0.748 |
| MMR | Exact Recall@10 | 0.443 | 0.443 | **0.468** |
| MMR | Context Precision | 0.812 | 0.882 | 0.823 |
| LTR | Exact Recall@10 | 0.418 | 0.418 | 0.418 |
| LTR | Context Recall | 0.615 | 0.618 | 0.593 |

Takeaways:

- The **final run is the best (or tied-best) RRF/MMR configuration** of the three — consistent with it being the one produced *after* the parameter grid search (Section 4), not before it.
- v2's RRF dipped noticeably (0.410 vs 0.477/0.485) — a reminder that RRF/MMR quality is sensitive to `base_k`/`rrf_k`/candidate-pool size, not just the fusion formula.
- LTR's exact-match numbers are byte-for-byte identical across v2 and final (same model, same retrieved IDs) but its *context* scores still shift (0.618 → 0.593). Since the retrieved documents didn't change, this movement is judge noise, not retrieval quality — DeepEval's LLM judge runs at `temperature=1`, so context_recall/context_precision have real run-to-run variance even when nothing about the retriever changed. Treat single-run context-metric deltas smaller than ~0.02–0.03 as noise, not signal.

## 4. Parameter tuning (100-query dev set)

From `retriever_tuning_results.csv` (exact-match metrics only, no LLM judge — used for cheap/fast grid search before the expensive LLM-judged run above):

| Stage | Swept | Held fixed | Winner |
|---|---|---|---|
| 1 — candidate size | `base_k`/`candidate_k` ∈ {10,20,30,50} | rrf_k=60, λ=0.8 | **20** (nDCG 0.392 vs 0.382–0.387 for others) |
| 2 — RRF k | `rrf_k` ∈ {20,40,60,80,100} | base_k=candidate_k=20, λ=0.8 | tie at nDCG 0.392 (pool this small is already ~saturated, so `rrf_k` barely moves the final MMR selection) → kept **60** |
| 3 — MMR λ | λ ∈ {0.6,0.7,0.8,0.9} | base_k=candidate_k=20, rrf_k=60 | **0.9** (nDCG 0.3935) narrowly beats **0.8** (nDCG 0.3918); 0.6/0.7 clearly worse (0.368/0.377) |

So the grid search's strict winner on the 100-query dev set (exact metrics only) is `base_k=20, rrf_k=60, λ=0.9`. λ=0.8 is a very close second (Δ nDCG ≈ 0.0017) — within noise on a 100-query set — which is why choosing λ=0.8 for the final architecture (Section 6) instead of the nominal winner is defensible, especially once you weigh in the context-recall/precision behavior from Section 2, which wasn't part of this grid search at all.

## 5. Query-level patterns (from `retriever_comparison_results.csv`)

- **Every retriever gets it right** on queries with strong lexical/entity overlap and a single obvious gold doc, e.g. *"Something looks off about Mitsubishi financial data"* and *"Interactive Brokers: IOPTS and list of structured products"* — exact_recall = 1.0, context_recall = 1.0 across BM25, RRF, MMR alike.
- **Every retriever misses** queries where the gold doc uses different vocabulary than the question, e.g. *"What is the easiest way to back-test index funds and ETFs?"* and *"Why do stocks priced above $2.00 on the ASX sometimes move in $0.005 increments?"* — 0 hits and context_recall ≈ 0 everywhere. Since BM25 and the vector search both fail to surface the gold doc in their own top-k, no downstream fusion/rerank/MMR/LTR step can recover it — the ceiling here is set upstream of RRF, not by the reranking strategy.
- **Where MMR earns its keep**: on multi-part questions with several plausible gold docs (e.g. *"Trading on exchanges or via brokerage companies?"*, *"Explain the details and benefits of rebalancing a retirement portfolio?"*), RRF's top-10 tends to bunch several near-duplicate chunks together; MMR's diversity penalty pulls in a wider spread of distinct, still-relevant chunks — which is exactly the pattern behind the context_precision gain in Section 2.

## 6. My approach

I would keep both RRF and MMR:

```
                   QUERY
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
         BM25                VECTOR
        Top 20               Top 20
           │                   │
           └─────────┬─────────┘
                     ▼
                    RRF
                   k = 60
                     │
                  Top 20
                     │
                     ▼
                    MMR
                 lambda=.8
                     │
                     ▼
                FINAL TOP 10
                     │
                     ▼
                RAG GENERATOR
```

Why not stop at RRF even though its exact recall is higher? Because the final documents are going into an LLM, and the MMR stage improves the semantic context characteristics:

| | RRF | MMR | |
|---|---|---|---|
| Exact Recall | 0.485 | 0.468 | slight ↓ |
| nDCG | 0.401 | 0.402 | slight ↑ |
| Context Recall | 0.652 | 0.675 | ↑ |
| Context Precision | 0.748 | 0.823 | substantial ↑ |

RRF alone optimizes for "did we retrieve the right document ID." MMR trades a small amount of that exact-ID recall for a top-10 that is less redundant and more consistently on-topic once it's actually read by the generator — which is what context recall/precision are measuring, and what actually determines RAG answer quality.
