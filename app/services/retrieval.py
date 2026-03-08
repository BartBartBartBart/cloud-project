from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import Any

model = OllamaLLM(model="qwen2.5:1.5b", num_gpu=0)


PROMPT_TEMPLATE = """<|user|>
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough
information to answer this."

Context:
{context}

Question:
{question}
<|end|>
<|assistant|>"""


def retrieve_and_generate(vectorstore: FAISS, query: str, k: int = 3) -> dict[str, Any]:
    """
    Retrieve relevant documents from the vectorstore and
    generate an answer to the query.
    """
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n---\n\n".join(
        [f"{result.page_content} (metadata: {result.metadata})" for result in results]
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | model
    response = chain.invoke({"context": context, "question": query})
    return {
        "response": response,
        "retrieved_chunks": results,
    }
