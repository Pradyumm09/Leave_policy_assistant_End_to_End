from crewai.tools import tool

from app.llamaindex.retriever import LeavePolicyRetriever


retriever = LeavePolicyRetriever(top_k=5)


@tool("Leave Policy Search")
def leave_policy_search(query: str) -> str:
    """
    Search the official employee leave policy.
    Use this tool for leave-policy questions.
    """

    results = retriever.retrieve(query)

    if not results:
        return "No relevant leave policy information was found."

    output = []

    for result in results:
        output.append(
            f"""
Document: {result['metadata'].get('document_id', 'Unknown')}
Section: {result['metadata'].get('section', 'Unknown')}

{result['content']}
"""
        )

    return "\n---\n".join(output)