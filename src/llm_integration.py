# integrates with hugging face llm to answer questions and measure response time
from huggingface_hub import InferenceClient
from src.embeddings import search_faiss_index
from dotenv import load_dotenv
import os
import time

load_dotenv()

# load hugging face token from env variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")

# initialize hugging face inference client
client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACE_TOKEN,
)

data_path = "data/cleaned_hotel_bookings.csv"
index_path = "models/bookings_index.faiss"


# queries the index and sends result to hugging face
def query_huggingface_api(query, top_k=50):
    # retrieve relevant context
    results = search_faiss_index(query, index_path, data_path, top_k=top_k)
    context = " ".join(str(r) for r in results)

    # prompt
    messages = [
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query}\nProvide only the direct answer without explanation, calculations, or code:"
        }
    ]
    
    # query the model
    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=500,
    )
    return completion.choices[0].message.content


# measures and prints response time
def measure_api_performance(query):
    start_time = time.time()
    response = query_huggingface_api(query)
    total_time = time.time() - start_time
    print(f"API Response Time: {total_time:.4f} seconds")
    return response


if __name__ == "__main__":
    while True:
        query = input("Enter your question (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Chat End.")
            break
        answer = measure_api_performance(query)
        print("Generated Answer:", answer)