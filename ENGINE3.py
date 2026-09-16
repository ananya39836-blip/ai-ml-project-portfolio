import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from flask import Flask, request, jsonify
import redis
import time

# -------------------------------S
# Step 1: Load Larger Dataset
# -------------------------------
# Example: load from CSV
ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4],
    'item_id': [101, 102, 101, 103, 104, 105, 101],
    'rating': [5, 3, 4, 2, 5, 4, 3]
})
 # user_id, item_id, rating
items = pd.DataFrame({
    'item_id': [101, 102, 103, 104, 105],
    'title': ["The Matrix", "Titanic", "Inception", "Avengers", "Interstellar"],
    'tags': ["sci-fi action cyberpunk",
             "romance drama historical",
             "sci-fi thriller dream",
             "superhero action marvel",
             "sci-fi space drama"]
})
 # item_id, title, genre, tags

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

# -------------------------------
# Step 2: Collaborative Filtering
# -------------------------------
cf_model = SVD()
cf_model.fit(trainset)

# -------------------------------
# Step 3: Content-Based Filtering
# -------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')
item_profiles = tfidf.fit_transform(items['tags'].fillna(""))

def content_recommend(item_id, top_n=5):
    idx = items[items['item_id'] == item_id].index[0]
    sim_scores = cosine_similarity(item_profiles[idx], item_profiles).flatten()
    top_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    return items.iloc[top_indices]['item_id'].tolist()

# -------------------------------
# Step 4: Hybrid Recommendation
# -------------------------------
def hybrid_recommend(user_id, top_n=5):
    # Collaborative predictions
    all_items = items['item_id'].unique()
    user_items = ratings[ratings['user_id'] == user_id]['item_id'].unique()
    unseen_items = [iid for iid in all_items if iid not in user_items]

    predictions = [cf_model.predict(user_id, iid) for iid in unseen_items]
    predictions.sort(key=lambda x: x.est, reverse=True)
    cf_top = [int(pred.iid) for pred in predictions[:top_n]]

    # Content boost: add similar items to top CF item
    if cf_top:
        content_top = content_recommend(cf_top[0], top_n=top_n)
        hybrid = list(set(cf_top + content_top))
    else:
        hybrid = cf_top

    return hybrid[:top_n]

# -------------------------------
# Step 5: Redis Caching
# -------------------------------
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_recommend(user_id, top_n=5):
    cache_key = f"user:{user_id}:recs"
    cached = redis_client.get(cache_key)
    if cached:
        return eval(cached.decode("utf-8"))
    recs = hybrid_recommend(user_id, top_n)
    redis_client.set(cache_key, str(recs), ex=300)  # cache for 5 min
    return recs

# -------------------------------
# Step 6: Flask API
# -------------------------------
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.json['user_id'])
    start = time.time()
    recs = cached_recommend(user_id, top_n=5)
    latency = round((time.time() - start) * 1000, 2)
    return jsonify({"user_id": user_id, "recommendations": recs, "latency_ms": latency})

# -------------------------------
# Step 7: Benchmarking
# -------------------------------
def benchmark(users=[1,2,3,4,5]):
    latencies = []
    for u in users:
        start = time.time()
        _ = cached_recommend(u, top_n=5)
        latencies.append((time.time() - start) * 1000)
    print("Avg Latency (ms):", sum(latencies)/len(latencies))


def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / k
K = 5

def calculate_metrics():

    p5='Precision_at_5'
    r5='Recall@5'
    n5='NDCG@5'
    h5='Hit Rate'
    return p5,r5,n5,h5
print(calculate_metrics())
if __name__=='__main__':
    app.run(debug=True) 

