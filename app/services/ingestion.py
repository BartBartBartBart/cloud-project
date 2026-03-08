from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re
import os
import sys
import Any

from retrieval import retrieve_and_generate


def load_pdf(file_path: str) -> list[str]:

    reader = PdfReader(file_path)
    return [page.extract_text() for page in reader.pages]


def clean_text(text: str) -> str:
    """
    Clean page of text by removing extra newlines, spaces, and
    fixing broken lines.
    """
    # >2 \n -> \n\n
    text = re.sub(r"\n{3,}", "\n\n", text)
    # >1 space -> 1 space
    text = re.sub(r" {2,}", " ", text)
    # strip leading/trailing whitespace
    text = text.strip()

    # Fix broken lines
    # Split word across lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # \n not preceded or followed by punctuation
    text = re.sub(r"([^\n\.\!\?])\n([^\n\.\!\?])", r"\1 \2", text)
    # \n is followed by a lowercase letter
    text = re.sub(r"\n([a-z])", r" \1", text)

    # Remove page numbers
    # Line is only digits -> remove
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    # Line is "Page X of Y" -> remove
    text = re.sub(r"^Page \d+ of \d+$", "", text, flags=re.MULTILINE)

    return text


def chunk_text(
    text: list[str], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    For each page, clean and create a Document with page content
    and metadata. Then use RecursiveCharacterTextSplitter to create
    chunks of text from the list of Documents.
    """
    docs = []
    chunk_id = 0
    for i, page in enumerate(text):
        cleaned_page = clean_text(page)
        doc = Document(
            page_content=cleaned_page,
            metadata={
                "page_number": i + 1,
                "chunk_id": chunk_id,
                "source": sys.argv[1] if len(sys.argv) > 1 else "unknown",
            },
        )
        docs.append(doc)
        chunk_id += 1

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def create_embeddings(
    chunks: list[Document], model_name: str = "all-MiniLM-L12-v2"
) -> tuple[list[tuple[Any, Any, Any]], Any]:
    """
    Create embeddings for each chunk of text using the specified
    SentenceTransformer model. Returns a list of tuples containing
    the chunk text, its embedding vector, and the chunk metadata.
    We return the metadata here so it can later be stored in the
    vectorstore.
    """
    embedder = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": "cpu"}
    )
    embedding_vectors = embedder.embed_documents(
        [chunk.page_content for chunk in chunks]
    )
    embeddings = [
        (chunk.page_content, embedding_vectors[i], chunk.metadata)
        for i, chunk in enumerate(chunks)
    ]
    return embeddings, embedder


def create_vectorstore(
    embeddings: list[tuple[str, list[float], dict]], embedding_function=None
) -> FAISS:
    """
    Create a FAISS vector store using either precomputed embeddings or by
    letting LangChain compute them.  The input is a list of (text, vector,
    metadata) triples that were produced by :func:`create_embeddings`.
    """
    # Expected inputs for vectorstore constructor
    texts_and_vecs = [(text, vec) for text, vec, _ in embeddings]
    metadatas = [md for _, _, md in embeddings]
    return FAISS.from_embeddings(
        texts_and_vecs,
        embedding_function,
        metadatas=metadatas,
    )


def main():
    file_path = str(sys.argv[1]) if len(sys.argv) > 1 else None
    if (
        file_path is not None
        and os.path.isfile(file_path)
        and file_path.endswith(".pdf")
    ):
        # Load PDF and extract text from each page
        pdf_text = load_pdf(file_path)
        chunks = chunk_text(pdf_text)
        print(f"Created {len(chunks)} chunks.")

        # Create embeddings for each chunk of text
        embeddings, embedding_function = create_embeddings(chunks)
        print(f"Created embeddings for {len(embeddings)} chunks.")

        # Create a FAISS vector store from the embeddings
        vectorstore = create_vectorstore(embeddings, embedding_function)
        print("Created FAISS vector store.")

        # Retrieve relevant documents and generate an answer to the query
        results = retrieve_and_generate(
            vectorstore, "What is a letter-string analogy task?", k=3
        )
        print("Retrieved context for query:")
        for i, result in enumerate(results["retrieved_chunks"]):
            print(f"Result {i+1} [{result.metadata}]: {result.page_content}")
        print("\nGenerated response:")
        print(results["response"])

    else:
        print("Invalid file path or not a PDF file. Please try again.")


if __name__ == "__main__":
    main()
