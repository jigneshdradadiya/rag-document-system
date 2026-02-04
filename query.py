"""Query and retrieval module using LangChain."""
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL, INDEX_DIR, TOP_K
)


class VectorRetriever:
    """Handles vector search and retrieval using LangChain FAISS."""

    def __init__(self):
        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.load_index()

    def load_index(self) -> None:
        """Load FAISS vector store."""
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Index not found at {INDEX_DIR}. "
                "Please run ingest.py first to build the index."
            )

        print(f"Loading vector store from {INDEX_DIR}...")
        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"[OK] Vector store loaded successfully")

    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: User question
            k: Number of chunks to retrieve

        Returns:
            List of LangChain Documents with metadata
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def retrieve_with_scores(self, query: str, k: int = TOP_K) -> List[tuple]:
        """
        Retrieve top-k most relevant chunks with similarity scores.

        Args:
            query: User question
            k: Number of chunks to retrieve

        Returns:
            List of tuples (Document, score)
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def format_results(self, results: List[Document]) -> str:
        """Format retrieval results for display."""
        output = []
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            output.append(f"\n{'='*60}")
            output.append(f"Result {i} - [{metadata.get('filename', 'Unknown')} p.{metadata.get('page', '?')}]")
            output.append(f"{'-'*60}")
            content = doc.page_content
            output.append(content[:300] + "..." if len(content) > 300 else content)
        return "\n".join(output)


def main():
    """Test retrieval with example queries."""
    retriever = VectorRetriever()

    print("\n" + "="*60)
    print("Document QA - Retrieval Test")
    print("="*60)

    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        print(f"\nSearching for: '{query}'")
        results = retriever.retrieve(query)
        print(retriever.format_results(results))

        print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
