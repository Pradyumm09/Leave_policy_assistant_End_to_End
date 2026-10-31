from app.core.config import BASE_DIR, CONFIG
from app.ingestion.embedder import create_embedding_model
from app.ingestion.faiss_manager import load_faiss_index


def main():

    embeddings = create_embedding_model()

    vector_store_path = (
        BASE_DIR
        / CONFIG["paths"]["vector_store"]
    )

    vector_store = load_faiss_index(
        embeddings,
        vector_store_path,
    )

    question = "How many paternity leave days are allowed?"

    results = vector_store.similarity_search(
        question,
        k=5,
    )

    print("\n" + "=" * 70)
    print("QUERY:")
    print(question)
    print("=" * 70)

    for i, doc in enumerate(results, start=1):

        print(f"\nRESULT {i}")
        print("-" * 70)

        print("ID      :", doc.metadata.get("id"))
        print("Section :", doc.metadata.get("section"))
        print("Type    :", doc.metadata.get("type"))

        print("\nContent:")
        print(doc.page_content)


if __name__ == "__main__":
    main()
