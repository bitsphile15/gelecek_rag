from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient

# === FastAPI App ===
app = FastAPI()

# === Config ===
qdrant_url = "https://ae3eaacf-299d-4931-9081-3f1679c72635.eu-west-1-0.aws.cloud.qdrant.io:6333"
qdrant_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.R7kudU92gPD3HrWOXSzYh5MTCGcFm5i03CaIK0PQ5A4"
collection_name = "Gelecke_rag"

# === Load Embedding Model ===
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# === Initialize Qdrant Client ===
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_key,
)

# === Load Vector Store ===
qdrant = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings
)

# === Load Chat Model ===
model = ChatGroq(
    model="llama3-70b-8192",
    api_key="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",
    temperature=0,
    max_tokens=1024,
    timeout=30,
    max_retries=2
)

# === Query Prompt ===
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Generate 5 query variations for: {question}"""
)

# === Retriever ===
retriever = MultiQueryRetriever.from_llm(
    qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    ),
    llm=model,
    prompt=QUERY_PROMPT
)

# === Main Prompt ===
final_prompt = ChatPromptTemplate.from_template("""
Answer using ONLY this context:
{context}

Question: {question}

If no relevant information, say "I cannot respond to that."
""")

# === Helper to format docs ===
def format_doc(doc):
    return f"[Source: {doc.metadata.get('source', '?')}]\n{doc.page_content}"

# === Chain ===
chain = (
    RunnableParallel({
        "context": retriever | (lambda docs: "\n\n".join(format_doc(d) for d in docs)),
        "question": RunnablePassthrough()
    })
    | final_prompt
    | model
)

# === Request body schema ===
class QueryRequest(BaseModel):
    question: str

# === API Endpoint ===
@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        result = chain.invoke(request.question)
        return {"answer": result.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
