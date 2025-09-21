import argparse
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "/usr/mlflow/data"

def main(top_k):
    anime = pd.read_csv(os.path.join(DATA_DIR, "anime_clean.csv"))
    ratings_train = pd.read_csv(os.path.join(DATA_DIR, "ratings_train.csv"))
    ratings_test = pd.read_csv(os.path.join(DATA_DIR, "ratings_test.csv"))

    # 建立 user-item 矩陣
    user_item_matrix = ratings_train.pivot_table(
        index="user_id", columns="anime_id", values="rating"
    ).fillna(0)

    # 建立 KNN 模型
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6, n_jobs=-1)
    knn.fit(user_item_matrix)

    sample_users = np.random.choice(ratings_train["user_id"].unique(), 50, replace=False)

    precisions, recalls = [], []
    rec_records = []

    for u in sample_users[:5]:
        if u not in user_item_matrix.index:
            continue

        # 找相似使用者
        user_vector = user_item_matrix.loc[[u]]
        _, indices = knn.kneighbors(user_vector, n_neighbors=6)
        neighbor_ids = user_item_matrix.index[indices.flatten()[1:]]  # 排除自己

        neighbor_ratings = user_item_matrix.loc[neighbor_ids]
        mean_scores = neighbor_ratings.mean().sort_values(ascending=False)

        # 過濾已看過
        seen = user_item_matrix.loc[u]
        seen = seen[seen > 0].index
        rec_ids = mean_scores.drop(seen).head(top_k).index

        recs = set(rec_ids)
        user_test = ratings_test[ratings_test["user_id"] == u]
        liked = set(user_test[user_test["rating"] > 7]["anime_id"])
        if len(liked) == 0:
            continue

        hit = len(recs & liked)
        precisions.append(hit / top_k)
        recalls.append(hit / len(liked))

        rec_records.append({
            "user_id": u,
            "liked_in_test": anime[anime["anime_id"].isin(liked)]["name"].tolist(),
            "recommended": anime[anime["anime_id"].isin(rec_ids)][["anime_id", "name"]].to_dict(orient="records")
        })

    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0

    # ===== MLflow logging =====
    mlflow.log_param("model", "user_based_cf")
    mlflow.log_param("sample_users", 50)
    mlflow.log_param("top_k", top_k)
    mlflow.log_metric("precision_at_10", mean_precision)
    mlflow.log_metric("recall_at_10", mean_recall)

    # 輸出推薦清單 CSV
    df_examples = pd.DataFrame(rec_records)
    out_path = os.path.join(DATA_DIR, "user_based_examples.csv")
    df_examples.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path, artifact_path="recommendations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    main(args.top_k)
