
# import pandas as pd
# import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss
# import uvicorn

# # Step 1: Load Dataset (sample 1000 rows for performance)
# df = pd.read_csv("Hotel_Reviews.csv")
# df = df[["Hotel_Name", "Hotel_Address"]].dropna().head(1000)

# # Combine data to create a search field
# df["content"] = df["Hotel_Name"] + ". " + df["Hotel_Address"] 

# # Step 2: Generate Embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)

# # Step 3: Create FAISS Index
# dimension = embeddings[0].shape[0]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(embeddings))

# # Step 4: FastAPI Setup
# app = FastAPI()

# class SearchRequest(BaseModel):
#     query: str
#     top_k: int = 5

# @app.post("/search")
# def search_hotels(req: SearchRequest):
#     query_vector = model.encode([req.query])
#     distances, indices = index.search(np.array(query_vector), req.top_k)
#     results = df.iloc[indices[0]][["Hotel_Name", "Hotel_Address"]].to_dict(orient="records")
#     return {"results": results}

# # Step 5: Run app
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)



import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import uvicorn

from geopy.distance import geodesic
df = pd.read_csv("Hotel_Reviews.csv")
df = df[["Hotel_Name", "Hotel_Address","Positive_Review","lat","lng"]].dropna().head(1000)

df["content"] = df["Hotel_Name"] + ". " + df["Hotel_Address"] + ". " + df["Positive_Review"] + ". " + df["lat"].astype(str) + ". " + df["lng"].astype(str)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

# FAISS Index for Cosine Similarity
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product Index
index.add(embeddings)

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_hotels(req: SearchRequest):
    query_lower = req.query.strip().lower()
    # print("query_lower",query_lower)
    # print(df["Hotel_Name"].str.lower())

    exact_match_df = df[df["Hotel_Name"].str.lower() == query_lower]
    exact_results = []
    if not exact_match_df.empty:
        exact_results = exact_match_df[["Hotel_Name", "Hotel_Address"]].copy()
        exact_results["Similarity_Score"] = 1.0

    # Semantic fallback
    query_vector = model.encode([req.query]).astype("float32")
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, req.top_k)

    semantic_results = []
    for idx, score in zip(indices[0], distances[0]):
        hotel = df.iloc[idx]
        semantic_results.append({
            "Hotel_Name": hotel["Hotel_Name"],
            "Hotel_Address": hotel["Hotel_Address"],
            "Similarity_Score": round(float(score), 4)
        })

    # Combine and deduplicate results
    combined = pd.DataFrame(exact_results).to_dict(orient="records") + semantic_results
    seen = set()
    final_results = []
    for r in combined:
        key = (r["Hotel_Name"], r["Hotel_Address"])
        if key not in seen:
            final_results.append(r)
            seen.add(key)
        if len(final_results) == req.top_k:
            break

    return {"results": final_results}


class NearbyRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 3.0 

@app.post("/hotels-near-me")
def hotels_near_me(req: NearbyRequest):
    user_coords = (req.latitude, req.longitude)

    def calculate_distance(row):
        hotel_coords = (row["lat"], row["lng"])
        return geodesic(user_coords, hotel_coords).km

    df["distance_km"] = df.apply(calculate_distance, axis=1)

    nearby = df[df["distance_km"] <= req.radius_km].copy()
    nearby = nearby.sort_values("distance_km")

    results = nearby[["Hotel_Name", "Hotel_Address", "distance_km"]].to_dict(orient="records")
    return {"results": results}    


# Step 5: Run app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
