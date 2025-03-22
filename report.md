# InnSights: Project Report

## Project Overview
This project provides a hotel booking analytics system with a Retrieval-Augmented Generation (RAG) pipeline for answering booking-related queries. It integrates FAISS for efficient data retrieval, SentenceTransformer for embedding generation, and Hugging Face's Mistral-7B-Instruct-v0.3 for natural language understanding. FastAPI serves as the backend to handle API requests, enabling data-driven insights and automated question answering.

## Implementation Choices

### FAISS 
FAISS (Facebook AI Similarity Search) can handle large-scale similarity searches efficiently. It provides fast nearest-neighbor search over high-dimensional embeddings, and is ideal for retrieving relevant booking records quickly.

### Hugging Face's Mistral-7B-Instruct-v0.3 
Mistral-7B-Instruct-v0.3 was integrated for its strong performance in instruction-following tasks. It generates quick and coherent answers when provided with relevant booking context.

### FastAPI 
FastAPI was used to structure the backend due to its simplicity and speed. It promotes modular code design and is easy to maintain and extend.

## Challenges
1. **Optimizing FAISS Index**: Ensuring the FAISS index handled the datasets efficiently required tuning of embedding generation.
2. **Managing Environment Variables**: Sensitive API keys for Hugging Face were managed through a .env file to keep them secure.
3. **Response Time**: Generating embeddings and querying the LLM introduced latency, which required reducing redundant computations and selecting the right model.
4. **Excluding Large Files from GitHub**: The FAISS index file (bookings_index.faiss) exceeded GitHub’s file size limit of 100MB. Git LFS was initially implemented, but GitHub's size limitations persisted. As a workaround, the FAISS index file was excluded from the repository, and instructions were added to regenerate the index locally after cloning the repo.
