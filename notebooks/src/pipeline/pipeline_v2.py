import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow

DATA_DIR = "/usr/mlflow/data"

class AnimePipeline:
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size

    def load_data(self):
        """載入動畫資料，只取部分樣本確保 3 分鐘內可跑完"""
        anime = pd.read_csv(os.path.join(DATA_DIR, "anime_clean.csv"))
        ratings_train = pd.read_csv(os.path.join(DATA_DIR, "ratings_train.csv"))

        # 抽樣，避免跑全量太久
        anime = anime.sample(self.sample_size, random_state=42).reset_index(drop=True)
        return anime, ratings_train

    def train_model(self, anime, max_features=1000, ngram_range=(1,1), min_df=2, use_type=True):
        """用 TF-IDF 訓練 item-based 模型，可以選擇是否加入 type 特徵"""

        # ✅ 拼接 genre + type 作為新的特徵
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

    def evaluate_and_log(self, anime, sim_matrix, params):
        """簡單評估 Precision@10 並 log 到 MLflow"""

        def precision_at_k(recommended, relevant, k=10):
            return len(set(recommended[:k]) & set(relevant)) / k

        # ✅ 抽樣 30 部動畫做測試，加快速度
        test_idx = np.random.choice(len(anime), 30, replace=False)
        scores = []
        for idx in test_idx:
            sim_scores = list(enumerate(sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_idx = [i for i, _ in sim_scores[1:11]]
            recommended = anime.iloc[top_idx]["name"].tolist()
            relevant = anime[anime["genre"] == anime.iloc[idx]["genre"]]["name"].tolist()
            if len(relevant) > 1:
                scores.append(precision_at_k(recommended, relevant, k=10))

        avg_precision = np.mean(scores)

        # 👉 mlflow.start_run(): 開始一個新的實驗 run
        with mlflow.start_run(run_name="pipeline-tfidf") as run:
            mlflow.log_params(params)   # 記錄參數
            mlflow.log_metric("precision_at_10", avg_precision)  # 記錄指標

            print("Run ID:", run.info.run_id)
            print("Artifact URI:", run.info.artifact_uri)

        return avg_precision
