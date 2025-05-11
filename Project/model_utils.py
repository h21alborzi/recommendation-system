import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_model_and_index():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df = pd.read_csv("products.csv")
    df["text"] = df["Name"] + " - " + df["Type"] + " - " + df["Gender"] + " - " + df["Material"] +  " - " +df["Brand"] +  " - " + str(df["Price"])

    # Embeddings
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index, df

# Search logic
def search_products(query, model, index, df, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb)
    D, I = index.search(q_emb, top_k)

    results = []
    for idx in I[0]:
        item = df.iloc[idx]
        results.append({
            "title": item["Name"],
            "brand": item["Brand"],
            "id": item["Page_number"],
            "type": item["Type"],
            "score": float(D[0][list(I[0]).index(idx)])
        })
    print(results)
    return results
