from app.llamaindex.retriever import LeavePolicyRetriever


retriever = LeavePolicyRetriever(top_k=5)


query = "How many days of paternity leave are allowed?"


results = retriever.retrieve(query)


print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(query)


print("\n" + "=" * 70)
print("RETRIEVED RESULTS")
print("=" * 70)


for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("Score:", result["score"])

    print(
        "Document ID:",
        result["metadata"].get("document_id"),
    )

    print(
        "Section:",
        result["metadata"].get("section"),
    )

    print(
        "Content:",
        result["content"],
    )