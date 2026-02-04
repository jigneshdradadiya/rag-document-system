"""Document ingestion and indexing module using LangChain."""
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

from config import (
    DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, SUPPORTED_EXTENSIONS
)


class DocumentIngestor:
    """Handles document loading, chunking, and indexing using LangChain."""

    def __init__(self):
        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.documents: List[Document] = []

    def load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF and extract text per page as LangChain Documents."""
        docs = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "page": page_num,
                            "doc_id": file_path.stem
                        }
                    )
                    docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return docs

    def load_text_file(self, file_path: Path) -> List[Document]:
        """Load plain text or markdown file as LangChain Document."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "page": 1,
                    "doc_id": file_path.stem
                }
            )
            return [doc]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_documents(self) -> None:
        """Load all supported documents from data directory."""
        print(f"Loading documents from {DATA_DIR}...")
        self.documents = []

        for file_path in DATA_DIR.rglob("*"):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                print(f"Loading: {file_path.name}")
                if file_path.suffix.lower() == ".pdf":
                    self.documents.extend(self.load_pdf(file_path))
                else:
                    self.documents.extend(self.load_text_file(file_path))

        print(f"Loaded {len(self.documents)} document pages/sections")

    def build_index(self) -> FAISS:
        """Build and save FAISS vector store using LangChain."""
        if not self.documents:
            print("No documents to index. Run load_documents() first.")
            return None

        # Split documents into chunks
        print("Chunking documents...")
        chunks = []
        for doc in tqdm(self.documents, desc="Splitting"):
            doc_chunks = self.text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)

        print(f"Created {len(chunks)} chunks")

        # Create FAISS vector store
        print("Building FAISS vector store with embeddings...")
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Save vector store
        print(f"Saving vector store to {INDEX_DIR}...")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))

        print(f"[OK] Index built successfully!")
        print(f"  - {len(chunks)} chunks indexed")
        print(f"  - Saved to: {INDEX_DIR}")

        return vectorstore


def main():
    """Main ingestion pipeline."""
    print("="*60)
    print("Document Ingestion Pipeline (LangChain)")
    print("="*60)

    ingestor = DocumentIngestor()

    # Load documents
    ingestor.load_documents()

    if not ingestor.documents:
        print("\n[WARNING] No documents found in data/ directory!")
        print("Please add PDF, TXT, or MD files to the data/ directory.")
        return

    # Build vector store
    vectorstore = ingestor.build_index()

    if vectorstore:
        print("\n[OK] Ingestion complete! You can now:")
        print("  - Run: python chat.py")
        print("  - Run: streamlit run app.py")


if __name__ == "__main__":
    main()
