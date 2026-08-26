from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline
)
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# 1. Load Embedding Model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load Chroma Database

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# 3. User Query


query = "How much did Microsoft pay to acquire GitHub?"


# 4. Retrieve Relevant Documents


retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)


print(f"\nUser Query: {query}")
print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)


# 5. Create RAG Prompt


context = "\n\n".join(
    [doc.page_content for doc in relevant_docs]
)

prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the information provided in the context.

If the answer cannot be found in the context, respond exactly:

I don't have enough information to answer that question based on the provided documents.

Context:
{context}

Question:
{query}

Answer:
"""


# 6. Load Free Local Hugging Face Model


print("\nLoading language model...")

hf_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    do_sample=False
)

model = HuggingFacePipeline(
    pipeline=hf_pipeline
)

# 7. Generate Answer


print("Generating answer...")

result = model.invoke(prompt)


# 8. Display Answer

print("\n--- Generated Response ---")
print(result)