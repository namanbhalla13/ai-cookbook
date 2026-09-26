"""
Vector search over the full FiQA corpus using:

- OpenAI embeddings
- ChromaDB persistent storage

Behavior:
1. If documents are missing from ChromaDB:
      embed only the missing documents
2. If the database is already complete:
      skip corpus embedding
3. For every search:
      embed only the query
4. Return top-k corpus IDs + text
"""

from pathlib import Path
import json

import chromadb
from openai import OpenAI
from dotenv import load_dotenv


# ==============================================================
# LOAD ENVIRONMENT
# ==============================================================

load_dotenv()


# ==============================================================
# CONFIG
# ==============================================================

DATA_DIR = Path(__file__).parent / "fiqa"

CORPUS_PATH = DATA_DIR / "full_corpus.json"

CHROMA_PATH = DATA_DIR / "chroma_db"

COLLECTION_NAME = "fiqa_corpus"

EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 10

# Number of documents sent in each OpenAI embedding request
EMBEDDING_BATCH_SIZE = 100


# ==============================================================
# OPENAI CLIENT
# ==============================================================

openai_client = OpenAI()


# ==============================================================
# CHROMA CLIENT
# ==============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# ==============================================================
# CHROMA COLLECTION
# ==============================================================

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)


# ==============================================================
# EMBEDDING FUNCTION
# ==============================================================

def create_embeddings(texts):
    """
    Create OpenAI embeddings for a list of strings.
    """

    if not texts:
        return []

    # Extra protection against empty input
    cleaned_texts = []

    for text in texts:

        if text is None:
            text = ""

        text = str(text).strip()

        if not text:
            text = "[EMPTY DOCUMENT]"

        cleaned_texts.append(text)


    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned_texts,
    )


    return [
        item.embedding
        for item in response.data
    ]


# ==============================================================
# LOAD FULL CORPUS
# ==============================================================

def load_corpus():

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        corpus = json.load(f)

    return corpus


# ==============================================================
# BUILD / RESUME VECTOR DATABASE
# ==============================================================

def build_vector_database():

    corpus = load_corpus()

    total_documents = len(corpus)

    print(
        f"Corpus documents: "
        f"{total_documents}"
    )


    for start in range(
        0,
        total_documents,
        EMBEDDING_BATCH_SIZE,
    ):

        end = min(
            start + EMBEDDING_BATCH_SIZE,
            total_documents,
        )

        batch = corpus[start:end]


        # ------------------------------------------------------
        # Get all corpus IDs in this batch
        # ------------------------------------------------------

        batch_ids = [
            str(doc["corpus_id"])
            for doc in batch
        ]


        # ------------------------------------------------------
        # Check which IDs already exist in Chroma
        # ------------------------------------------------------

        existing = collection.get(
            ids=batch_ids,
            include=[]
        )

        existing_ids = set(
            existing["ids"]
        )


        # ------------------------------------------------------
        # Prepare only missing documents
        # ------------------------------------------------------

        ids = []
        texts = []
        metadatas = []


        for doc in batch:

            corpus_id = str(
                doc["corpus_id"]
            )


            # Already stored in Chroma
            if corpus_id in existing_ids:
                continue


            title = str(
                doc.get("title", "")
                or ""
            ).strip()


            text = str(
                doc.get("text", "")
                or ""
            ).strip()


            combined_text = (
                f"{title}\n{text}".strip()
            )


            # --------------------------------------------------
            # Skip completely empty documents
            # --------------------------------------------------

            if not combined_text:

                print(
                    f"Skipping empty document: "
                    f"{corpus_id}"
                )

                continue


            ids.append(
                corpus_id
            )

            texts.append(
                combined_text
            )

            metadatas.append(
                {
                    "corpus_id": corpus_id,
                    "title": title,
                }
            )


        # ------------------------------------------------------
        # Nothing new in this batch
        # ------------------------------------------------------

        if not texts:

            print(
                f"Processed "
                f"{end}/{total_documents}"
            )

            continue


        # ------------------------------------------------------
        # Create embeddings
        # ------------------------------------------------------

        embeddings = create_embeddings(
            texts
        )


        # ------------------------------------------------------
        # Store in Chroma
        # ------------------------------------------------------

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )


        print(
            f"Indexed "
            f"{end}/{total_documents} "
            f"| New docs: {len(ids)}"
        )


    print(
        f"\nVector database indexing finished."
    )

    print(
        f"Documents currently in Chroma: "
        f"{collection.count()}"
    )


# ==============================================================
# INITIALIZE DATABASE
# ==============================================================

def initialize_database():

    corpus = load_corpus()

    expected_count = len(corpus)

    current_count = collection.count()


    print(
        f"ChromaDB contains "
        f"{current_count}/{expected_count} documents."
    )


    # ----------------------------------------------------------
    # Database already appears complete
    # ----------------------------------------------------------

    if current_count >= expected_count:

        print(
            "Vector database already complete."
        )

        print(
            "Skipping corpus embedding."
        )

        return


    # ----------------------------------------------------------
    # Database incomplete
    # ----------------------------------------------------------

    print(
        "Vector database incomplete."
    )

    print(
        "Embedding only missing documents..."
    )

    build_vector_database()


# ==============================================================
# VECTOR SEARCH
# ==============================================================

def search_vector(
    query: str,
    k: int = 10,
):
    """
    Search ChromaDB using OpenAI query embedding.

    Returns:
    [
        {
            "rank": 1,
            "corpus_id": "...",
            "score": ...,
            "title": "...",
            "text": "..."
        }
    ]
    """


    # ----------------------------------------------------------
    # Validate query
    # ----------------------------------------------------------

    query = str(query).strip()

    if not query:
        return []


    # ----------------------------------------------------------
    # Embed ONLY the query
    # ----------------------------------------------------------

    query_embedding = create_embeddings(
        [query]
    )[0]


    # ----------------------------------------------------------
    # Chroma vector search
    # ----------------------------------------------------------

    response = collection.query(
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


    # ----------------------------------------------------------
    # Build results
    # ----------------------------------------------------------

    results = []


    ids = response["ids"][0]

    documents = response["documents"][0]

    metadatas = response["metadatas"][0]

    distances = response["distances"][0]


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

        # Chroma cosine distance:
        # lower distance = more similar
        #
        # similarity ~= 1 - distance

        similarity = (
            1 - float(distance)
        )


        results.append(
            {
                "rank": rank,

                "corpus_id": str(
                    corpus_id
                ),

                "score": similarity,

                "title": (
                    metadata.get(
                        "title",
                        ""
                    )
                    if metadata
                    else ""
                ),

                "text": (
                    document
                    or ""
                ),
            }
        )


    return results


# ==============================================================
# INITIALIZE DATABASE ON IMPORT
# ==============================================================

initialize_database()


# ==============================================================
# INTERACTIVE SEARCH
# ==============================================================

if __name__ == "__main__":

    while True:

        query = input(
            "\nAsk a question "
            "(or type 'exit'): "
        ).strip()


        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:

            break


        if not query:

            continue


        results = search_vector(
            query=query,
            k=TOP_K,
        )


        print(
            f"\nTop {TOP_K} results:"
        )


        for result in results:

            print(
                f"\n"
                f"{result['rank']}. "
                f"Corpus ID: "
                f"{result['corpus_id']} | "
                f"Score: "
                f"{result['score']:.4f}"
            )


            if result["title"]:

                print(
                    f"Title: "
                    f"{result['title']}"
                )


            text_preview = (
                result["text"][:500]
                .replace("\n", " ")
            )


            print(
                f"Text: "
                f"{text_preview}"
            )
