import argparse
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = "/usr/mlflow/data"

def main(top_k):
    anime = pd.read_csv(os.path.join(DATA_DIR, "anime_clean.csv"))
    ratings_train = pd.read_csv(os.path.join(DATA_DIR, "ratings_train.csv"))
    ratings_test = pd.read_csv(os.path.join(DATA_DIR, "ratings_test.csv"))

    # 建立 TF-IDF
    anime["text"] = anime["genre"].fillna("") + " " + anime["type"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(anime["text"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(anime.index, index=anime["anime_id"]).drop_duplicates()

    sample_users = np.random.choice(ratings_train["user_id"].unique(), 50, replace=False)

    precisions, recalls = [], []
    rec_records = []

    for u in sample_users[:5]:
        user_ratings = ratings_train[ratings_train["user_id"] == u]
        liked = user_ratings[user_ratings["rating"] > 7]["anime_id"].tolist()
        if len(liked) == 0:
            continue

        sim_scores = np.zeros(cosine_sim.shape[0])
        for anime_id in liked:
            if anime_id in indices:
                idx = indices[anime_id]
                sim_scores += cosine_sim[idx]

        sim_scores = sim_scores / len(liked)
        sim_indices = sim_scores.argsort()[::-1]

        seen = set(user_ratings["anime_id"])
        rec_ids = [anime.loc[i, "anime_id"] for i in sim_indices if anime.loc[i, "anime_id"] not in seen][:top_k]

        recs = set(rec_ids)
        user_test = ratings_test[ratings_test["user_id"] == u]
        liked_test = set(user_test[user_test["rating"] > 7]["anime_id"])
        if len(liked_test) == 0:
            continue

        hit = len(recs & liked_test)
        precisions.append(hit / top_k)
        recalls.append(hit / len(liked_test))

        rec_records.append({
            "user_id": u,
            "liked_in_test": anime[anime["anime_id"].isin(liked_test)]["name"].tolist(),
            "recommended": anime[anime["anime_id"].isin(rec_ids)][["anime_id", "name"]].to_dict(orient="records")
        })

    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0

    # ===== MLflow logging =====
    mlflow.log_param("model", "item_based_tfidf")
    mlflow.log_param("sample_users", 50)
    mlflow.log_param("top_k", top_k)
    mlflow.log_metric("precision_at_10", mean_precision)
    mlflow.log_metric("recall_at_10", mean_recall)

    # 輸出推薦清單 CSV
    df_examples = pd.DataFrame(rec_records)
    out_path = os.path.join(DATA_DIR, "item_based_examples.csv")
    df_examples.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path, artifact_path="recommendations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    main(args.top_k)
