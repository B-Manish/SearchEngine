import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import uvicorn

# Step 1: Load Dataset (sample 1000 rows for performance)
df = pd.read_csv("Hotel_Reviews.csv")
df = df[["Hotel_Name", "Hotel_Address"]].dropna().head(1000)

# Combine data to create a search field
df["content"] = df["Hotel_Name"] + ". " + df["Hotel_Address"] 

# Step 2: Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)

# Step 3: Create FAISS Index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 4: FastAPI Setup
app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_hotels(req: SearchRequest):
    query_vector = model.encode([req.query])
    distances, indices = index.search(np.array(query_vector), req.top_k)
    results = df.iloc[indices[0]][["Hotel_Name", "Hotel_Address"]].to_dict(orient="records")
    return {"results": results}

# Step 5: Run app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
