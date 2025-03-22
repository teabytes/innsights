# sets up a fastapi server to provide hotel booking analytics and handle user queries with rag
from fastapi import FastAPI, Request
from src.analytics import load_data, revenue_trends, cancellation_rate, geographical_distribution, lead_time_distribution
from src.embeddings import search_faiss_index
from src.llm_integration import query_huggingface_api
import os
import faiss
from huggingface_hub import InferenceClient


app = FastAPI()

# file paths
data_path = "data/cleaned_hotel_bookings.csv"
index_path = "models/bookings_index.faiss"

df = load_data(data_path)


# checks system status
@app.get("/health")
async def health_check():
    status = {"status": "healthy"}

    # check data file
    if not os.path.exists(data_path):
        status["data"] = "missing"
        status["status"] = "unhealthy"
    else:
        status["data"] = "available"

    # check faiss index
    try:
        faiss.read_index(index_path)
        status["faiss_index"] = "available"
    except Exception:
        status["faiss_index"] = "missing or corrupt"
        status["status"] = "unhealthy"

    # check huggingface api
    try:
        client = InferenceClient()
        client.get_model_info("mistralai/Mistral-7B-Instruct-v0.3")
        status["huggingface_api"] = "available"
    except Exception:
        status["huggingface_api"] = "unavailable"
        status["status"] = "unhealthy"

    return status


# returns all analytics as json
@app.post("/analytics")
async def get_analytics():
    analytics = {
        "revenue_trends": revenue_trends(df),
        "cancellation_rate": cancellation_rate(df),
        "geographical_distribution": geographical_distribution(df),
        "lead_time_distribution": lead_time_distribution(df)
    }
    return analytics


# answers booking-related questions using rag
@app.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        return {"error": "query is required"}
    
    # retrieve relevant records from index
    results = search_faiss_index(query, index_path, data_path, top_k=50)
    context = " ".join(str(r) for r in results)
    
    answer = query_huggingface_api(query)
    
    return {"query": query, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)