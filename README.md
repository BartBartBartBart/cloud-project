# A RAG-based Q&A LLM for PDF Files
A personal project to get familiar with techniques and tools such as LangChain, Docker, GCP and FastAPI. This project is capable of ingesting a PDF file, and using it for RAG-based querying of a LLM. It will soon be deployed on GCP.

The data flows is as follows: 
PDF → `ingestion.py` → FAISS vectorstore → `retrieval.py` → LLM → answer

## Tech Stack
The following libraries are used:
- LangChain, for ingesting the PDF, creating the vectorstore and embeddings, as well as the RAG-based querying. 
- Ollama, for a small LLM, integrated into LangChain.
- PyPDF, for reading the PDF file. 

## Project Structure 
The structure of the project is as follows:
```python
app/
    services/
        ingestion.py        # PDF chunking + vectorstore creation
        retrieval.py        # retrieval chain + LLM generation
scripts/
    test_rag.py             # Manual test
data/                       # place PDFs here
    lewis-robustness.pdf    # Sample paper
```

## Getting Started
First clone the repository and setup the conda evironment:
```bash
git clone https://github.com/BartBartBartBart/cloud-project.git
cd cloud-project
conda env create -f environment.yaml
conda activate rag-app
```
Then, install the project as a package and pull the qwen2.5:1.5b model. 
```bash
pip install -e .
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
ollama serve 
```
To run a full test of the RAG pipe, run:
```bash
python scripts/test_rag.py data/lewis-robustness.py
```

## Next Steps
Soon, the following milestones will be reached: 
- Wrapping in a FastAPI layer
- Docker Containerization
- Google Cloud Run deployment