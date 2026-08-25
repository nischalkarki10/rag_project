Conversational RAG Chatbot

A conversational Retrieval-Augmented Generation (RAG) chatbot that answers questions based on custom documents using LangChain, ChromaDB, Hugging Face embeddings, and a local Hugging Face language model.

The chatbot retrieves relevant information from a document collection and generates answers based only on the retrieved context. It also maintains conversation history, allowing follow-up questions to be understood in context.

Features
Document ingestion from .txt files
Document chunking using LangChain text splitters
Vector embeddings using sentence-transformers/all-MiniLM-L6-v2
Persistent vector storage using ChromaDB
Semantic document retrieval
Local answer generation using TinyLlama/TinyLlama-1.1B-Chat-v1.0
Conversational memory using chat history
Follow-up question rewriting for better document retrieval
No OpenAI API required
Runs locally with Hugging Face models