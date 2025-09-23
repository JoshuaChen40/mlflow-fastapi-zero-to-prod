import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow

DATA_DIR = "/usr/mlflow/data"

class AnimePipelineV4:
    def __init__(self, sample_size=500):
        # 抽樣確保速度快
        self.sample_size = sample_size

    def load_data(self):
        anime = pd.read_csv(os.path.join(DATA_DIR, "anime_clean.csv"))
        anime = anime.sample(self.sample_size, random_state=42).reset_index(drop=True)
        return anime

    def train_model(self, anime, max_features=500, ngram_range=(1,1), min_df=2, use_type=True):
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
        return sim_matrix, vectorizer

    def explain_and_log(self, anime, vectorizer, params):
        """同時輸出 單一樣本 + 全資料集 平均特徵重要性"""
        feature_names = vectorizer.get_feature_names_out()

        # ✅ 單一樣本 (第一筆動畫)
        tfidf_vector = vectorizer.transform([anime.iloc[0]["features"]]).toarray()[0]
        sample_importance = sorted(
            zip(feature_names, tfidf_vector),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        sample_dict = {f: float(v) for f, v in sample_importance}

        # ✅ 全資料集平均權重
        tfidf_matrix = vectorizer.transform(anime["features"])
        avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        global_importance = sorted(
            zip(feature_names, avg_weights),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        global_dict = {f: float(v) for f, v in global_importance}

        # ✅ 存到 MLflow
        with mlflow.start_run(run_name="pipeline-v4-explain") as run:
            mlflow.log_params(params)
            mlflow.log_dict(sample_dict, "sample_feature_importance.json")
            mlflow.log_dict(global_dict, "global_feature_importance.json")

            print("Run ID:", run.info.run_id)
            print("Artifacts URI:", run.info.artifact_uri)

        return sample_dict, global_dict
