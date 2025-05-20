import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sentence_transformers import SentenceTransformer

# Load data
items = pd.read_csv("products.csv")  # must have item_id
events = pd.read_csv("user_events.csv")  # must have user_id, item_id, event_type

# --- Step 1: Assign interaction scores ---
interaction_weights = {
    "view": 0.2,
    "click": 0.4,
    "add_to_cart": 0.7,
    "purchase": 1.0
}
events['score'] = events['event_type'].map(interaction_weights)
interactions = events.groupby(['user_id', 'item_id'])['score'].sum().reset_index()

# Normalize scores
scaler = MinMaxScaler()
interactions['interaction'] = scaler.fit_transform(interactions[['score']])
interactions.drop(columns=['score'], inplace=True)

# --- Step 2: Map users and items to integer indices ---
user_ids = interactions['user_id'].unique()
item_ids = items['item_id']

user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {i: j for j, i in enumerate(item_ids)}
rev_item_map = {j: i for i, j in item_map.items()}

print(rev_item_map)

interactions['user_idx'] = interactions['user_id'].map(user_map)
interactions['item_idx'] = interactions['item_id'].map(item_map)

# --- Step 3: Collaborative Filtering with Implicit ALS ---
user_item_matrix = coo_matrix(
    (interactions['interaction'], (interactions['user_idx'], interactions['item_idx']))
).tocsr()

cf_model = AlternatingLeastSquares(factors=64, iterations=15)
cf_model.fit(user_item_matrix)

# --- Step 4: Content-Based Filtering with TF-IDF ---
items['text'] = (
    items['name'].fillna('') + ' ' +
    items['type'].fillna('') + ' ' +
    items['gender'].fillna('') + ' ' +
    items['brand'].fillna('') + ' ' +
    items['material'].fillna('') + ' ' +
    pd.qcut(items['price'], q=5, labels=False).astype(str)
)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(items['text'])

# --- Step 5: Deep Learning Embeddings (SentenceTransformer) ---
model = SentenceTransformer('all-MiniLM-L6-v2')
item_embeddings = model.encode(items['text'].tolist(), normalize_embeddings=True)

# --- Step 6: Hybrid Recommendation Function ---
def hybrid_recommend(user_id, top_n=10, alpha=0.4, beta=0.3):
    if user_id not in user_map:
        return []

    user_idx = user_map[user_id]
    user_seen = set(user_item_matrix[user_idx].indices)

    # --- 6.1 Collaborative Filtering ---
    cf_recs = cf_model.recommend(user_idx, user_item_matrix[user_idx], N=top_n*2)
    print(cf_recs)
    items, scores = cf_recs
    cf_scores = dict(zip(items, scores))


    # --- 6.2 Content-Based (TF-IDF) ---
    cb_scores = {}
    for seen in user_seen:
        sims = cosine_similarity(tfidf_matrix[seen], tfidf_matrix).flatten()
        for idx, score in enumerate(sims):
            if idx not in user_seen:
                cb_scores[idx] = cb_scores.get(idx, 0) + score

    # --- 6.3 Deep Learning (Embedding Similarity) ---
    dl_scores = {}
    seen_embeddings = item_embeddings[list(user_seen)]
    sim_matrix = cosine_similarity(seen_embeddings, item_embeddings)
    for idx in range(sim_matrix.shape[1]):
        if idx not in user_seen:
            score = np.mean(sim_matrix[:, idx])
            dl_scores[idx] = dl_scores.get(idx, 0) + score

    # --- 6.4 Combine Scores ---
    all_items = set(cf_scores) | set(cb_scores) | set(dl_scores)
    final_scores = {}
    for item in all_items:
        cf = cf_scores.get(item, 0)
        cb = cb_scores.get(item, 0)
        dl = dl_scores.get(item, 0)
        final_scores[item] = alpha * cf + beta * cb + (1 - alpha - beta) * dl

    ranked_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [rev_item_map[i] for i, _ in ranked_items]


recommendations = hybrid_recommend(user_id=1, top_n=5)
recommended_items = items[items['item_id'].isin(recommendations)]
print(recommended_items[['item_id', 'name', 'brand', 'price']])
