from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceHubEmbeddings


# === Load environment variables from .env ===
load_dotenv()

# === FastAPI App ===
app = FastAPI()

# === Config ===
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("COLLECTION_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")
hugging_face_key=os.getenv("hg_key")


# Add CORS middleware configuration (same as first.py)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gelecek_rag.com"],  # Allowed origin
    allow_credentials=True,
    allow_methods=["*"],  # Allowed methods
    allow_headers=["*"],  # Allowed headers
)
# === Load Embedding Model ===

embeddings = HuggingFaceHubEmbeddings(
    repo_id="BAAI/bge-base-en-v1.5",
    huggingfacehub_api_token=hugging_face_key,
)

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
    api_key=groq_api_key,
    temperature=0,
    max_tokens=None,
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
