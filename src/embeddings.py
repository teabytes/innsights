# this script builds and searches a faiss index using sentence embeddings for hotel booking data
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os


# builds a faiss index from booking data
def build_faiss_index(data_path, index_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"file not found: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['reservation_status_date'])
    
    # embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # generate embeddings
    text_data = df.apply(lambda x: f"booking on {x['reservation_status_date']} at {x['hotel']}", axis=1)
    embeddings = np.array(model.encode(text_data.tolist(), batch_size=64, show_progress_bar=True))
    
    # create the index and add embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # save the index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"faiss index saved to {index_path}")


# searches the index for similar bookings
def search_faiss_index(query, index_path, data_path, top_k=5):
    # load index and data
    index = faiss.read_index(index_path)
    df = pd.read_csv(data_path, parse_dates=['reservation_status_date'])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # encode query and search index
    query_embedding = np.array(model.encode([query]))
    _, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]].to_dict(orient='records')
    return results


if __name__ == "__main__":
    data_path = "data/cleaned_hotel_bookings.csv"
    index_path = "models/bookings_index.faiss"

    build_faiss_index(data_path, index_path)
    
    # test search
    query = "booking on 2017-08-01 at city hotel"
    results = search_faiss_index(query, index_path, data_path)
    print("search results:", results)