import os
import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# === MLflow Tracking 設定 ===
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("anime-recsys-cicd")

client = MlflowClient()

# === 載入資料 ===
DATA_DIR = "/usr/mlflow/data"
anime = pd.read_csv(os.path.join(DATA_DIR, "anime_clean.csv"))
ratings_train = pd.read_csv(os.path.join(DATA_DIR, "ratings_train.csv"))

# === Step 1: 計算平均分數 ===
agg = (
    ratings_train.groupby("anime_id")["rating"]
    .mean()
    .reset_index()
    .merge(anime[["anime_id", "name"]], on="anime_id")
)

# === Step 2: 熱門前 7 ===
top7 = agg.sort_values("rating", ascending=False).head(7)

# === Step 3: 隨機選 3 部 ===
random_seed = np.random.randint(0, 100000)
np.random.seed(random_seed)
random3 = agg.sample(3)

# === Step 4: 組合 Top10 ===
top10 = pd.concat([top7, random3]).drop_duplicates("anime_id").head(10)
top10_ids = top10["anime_id"].tolist()
top10_names = top10["name"].tolist()
print(f"Random Seed: {random_seed}")
print("Top 10 Anime:", top10_names)

# === PopularTop10 模型定義 ===
class PopularTop10(mlflow.pyfunc.PythonModel):
    def __init__(self, anime_df, top10_ids):
        self.anime = anime_df
        self.top10_ids = top10_ids

    def predict(self, context, model_input):
        return [self.anime[self.anime["anime_id"].isin(self.top10_ids)]["name"].tolist()]

# === Step 5: Log + 註冊到 Registry ===
with mlflow.start_run(run_name="popular-top10-cron") as run:
    # Log params
    mlflow.log_param("model_type", "PopularTop10")
    mlflow.log_param("random_seed", random_seed)

    # Log artifact (Top10 JSON)
    result = {"random_seed": random_seed, "top10": top10_names}
    with open("top10.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    mlflow.log_artifact("top10.json")

    # 註冊模型
    result = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=PopularTop10(anime, top10_ids),
        registered_model_name="AnimeRecsysModel"
    )
    run_id = run.info.run_id

# === Step 6: Transition to Staging ===
latest_versions = client.get_latest_versions("AnimeRecsysModel", stages=["None"])
if latest_versions:
    new_version = max([int(v.version) for v in latest_versions])
    client.transition_model_version_stage(
        name="AnimeRecsysModel",
        version=new_version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"✅ AnimeRecsysModel v{new_version} 已自動設為 Staging")