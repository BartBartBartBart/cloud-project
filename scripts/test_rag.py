import os
import sys

from app.services.ingestion import (
    load_pdf,
    chunk_text,
    create_embeddings,
    create_vectorstore,
)
from app.services.retrieval import retrieve_and_generate

if __name__ == "__main__":
    file_path = str(sys.argv[1]) if len(sys.argv) > 1 else None
    if (
        file_path is not None
        and os.path.isfile(file_path)
        and file_path.endswith(".pdf")
    ):
        # Load PDF and extract text from each page
        print("-" * 8, " Creating chunks ", "-" * 8)
        pdf_text = load_pdf(file_path)
        chunks = chunk_text(pdf_text)
        print(f"Created {len(chunks)} chunks.")

        # Create embeddings for each chunk of text
        print("-" * 8, " Creating embeddings ", "-" * 8)
        embeddings, embedding_function = create_embeddings(chunks)
        print(f"Created embeddings for {len(embeddings)} chunks.")

        # Create a FAISS vector store from the embeddings
        print("-" * 8, " Creating vector store ", "-" * 8)
        vectorstore = create_vectorstore(embeddings, embedding_function)
        print("Created FAISS vector store.")

        # Retrieve relevant documents and generate an answer to the query
        print("-" * 8, " Retrieving and generating ", "-" * 8)
        results = retrieve_and_generate(
            vectorstore, "What is a letter-string analogy task?", k=3
        )
        print("Retrieved context for query:")
        for i, result in enumerate(results["retrieved_chunks"]):
            print(f"\nResult {i+1} [{result.metadata}]: {result.page_content}")
        print("\nGenerated response:")
        print(results["response"])
