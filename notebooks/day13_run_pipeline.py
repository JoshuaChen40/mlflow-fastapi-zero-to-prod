import mlflow
from src.pipeline.pipeline import AnimePipeline

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("anime-recsys-pipeline")

def main():
    pipeline = AnimePipeline(sample_size=1000)  # 控制資料量
    anime, ratings_train = pipeline.load_data()

    params = {
        "max_features": 1000,
        "ngram_range": (1,1),
        "min_df": 2
    }

    sim_matrix = pipeline.train_model(anime, **params)
    score = pipeline.evaluate_and_log(anime, sim_matrix, params)

    print(f"Pipeline 完成，Precision@10 = {score:.4f}")

if __name__ == "__main__":
    main()