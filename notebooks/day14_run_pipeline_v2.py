import mlflow
from src.pipeline.pipeline_v2 import AnimePipeline

# 👉 mlflow.set_tracking_uri(): 指定要連線的 MLflow Tracking Server
mlflow.set_tracking_uri("http://mlflow:5000")

# 👉 mlflow.set_experiment(): 指定實驗名稱，若不存在會自動建立
mlflow.set_experiment("anime-recsys-pipeline_v2")

def main():
    pipeline = AnimePipeline(sample_size=1000)
    anime, ratings_train = pipeline.load_data()

    # ✅ 可以調整 use_type=True/False，比較兩種特徵的差異
    params = {
        "max_features": 1000,
        "ngram_range": (1,1),
        "min_df": 2,
        "use_type": True
    }

    sim_matrix = pipeline.train_model(anime, **params)
    score = pipeline.evaluate_and_log(anime, sim_matrix, params)

    print(f"Pipeline 完成 ✅ Precision@10 = {score:.4f}")

if __name__ == "__main__":
    main()