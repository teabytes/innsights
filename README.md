# Innsights

Innsights is an LLM-Powered Booking Analytics and Q&A System system built to uncover insights from hotel reservation data. It combines data processing, machine learning, and natural language understanding to help analyze trends like revenue, cancellations, and guest origins. Using FAISS for fast similarity search and Hugging Face's Mistral-7B model for question answering, Innsights can quickly find patterns and answer booking-related questions. Designed with simplicity and efficiency in mind, it provides a solid foundation for exploring hotel data while being easy to expand with new features.


## Project Structure
```
project-root/
├── data/
│   ├── hotel_bookings.csv
│   └── cleaned_hotel_bookings.csv
├── models/
│   └── bookings_index.faiss
├── src/
│   ├── api.py
│   ├── analytics.py
│   ├── data_processing.py
│   ├── embeddings.py
│   └── llm_integration.py
├── README.md
└── requirements.txt
```

## Setup Instructions
1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the Hugging Face API token:
   ```bash
   export HUGGINGFACE_TOKEN=your_huggingface_api_token  # On macOS/Linux
   set HUGGINGFACE_TOKEN=your_huggingface_api_token     # On Windows
   ```

## Usage
1. Preprocess the data:
   ```bash
   python src/data_preprocessing.py
   ```
2. Build the FAISS index:
   ```bash
   python src/embeddings.py
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints
- `POST /analytics` - Returns various analytics about hotel bookings.
- `POST /ask` - Answers booking-related questions using Retrieval-Augmented Generation (RAG).
- `GET /health` - Checks system health by verifying data, FAISS index, and Hugging Face API.