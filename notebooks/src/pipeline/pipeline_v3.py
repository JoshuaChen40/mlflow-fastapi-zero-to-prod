import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow

DATA_DIR = "/usr/mlflow/data"

class AnimePipelineV3:
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size

    def load_data(self):
        """載入動畫資料，只取部分樣本確保 3 分鐘內可跑完"""
        anime = pd.read_csv(os.path.join(DATA_DIR, "anime_clean.csv"))
        ratings_train = pd.read_csv(os.path.join(DATA_DIR, "ratings_train.csv"))
        anime = anime.sample(self.sample_size, random_state=42).reset_index(drop=True)
        return anime, ratings_train

    def train_model(self, anime, max_features=1000, ngram_range=(1,1), min_df=2, use_type=True):
        """用 TF-IDF 訓練 item-based 模型，可以選擇是否加入 type 特徵"""
        if use_type:
            anime["features"] = anime["genre"].fillna("") + " " + anime["type"].fillna("")
        else:
            anime["features"] = anime["genre"].fillna("")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df
        )
        tfidf = vectorizer.fit_transform(anime["features"])
        sim_matrix = cosine_similarity(tfidf)
        return sim_matrix

    def predict(self, anime, sim_matrix, title, top_k=10):
        """統一推論格式"""
        if title not in anime["name"].values:
            return {"input": title, "recommendations": []}

        idx = anime[anime["name"] == title].index[0]
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_idx = [i for i, _ in sim_scores[1:top_k+1]]

        recs = anime.iloc[top_idx]["name"].tolist()
        return {"input": title, "recommendations": recs}

    def evaluate_and_log(self, anime, sim_matrix, params):
        """測試 Precision@10，並存推論範例到 MLflow artifacts"""

        def precision_at_k(recommended, relevant, k=10):
            return len(set(recommended[:k]) & set(relevant)) / k

        test_idx = np.random.choice(len(anime), 30, replace=False)
        scores = []
        examples = []

        for idx in test_idx[:5]:  # 只存 5 筆範例，避免 artifacts 太大
            sim_scores = list(enumerate(sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_idx = [i for i, _ in sim_scores[1:11]]
            recommended = anime.iloc[top_idx]["name"].tolist()
            relevant = anime[anime["genre"] == anime.iloc[idx]["genre"]]["name"].tolist()
            if len(relevant) > 1:
                scores.append(precision_at_k(recommended, relevant, k=10))

            # 存成統一格式
            examples.append({
                "input": anime.iloc[idx]["name"],
                "recommendations": recommended
            })

        avg_precision = np.mean(scores)

        with mlflow.start_run(run_name="pipeline-v3") as run:
            mlflow.log_params(params)
            mlflow.log_metric("precision_at_10", avg_precision)

            result_path = "recommendations.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(result_path)

            # mlflow.log_dict()，直接把 dict 存成 JSON
            mlflow.log_dict({"examples": examples}, "recommendations_dict.json")

            print("Run ID:", run.info.run_id)
            print("Artifact URI:", run.info.artifact_uri)

        return avg_precision