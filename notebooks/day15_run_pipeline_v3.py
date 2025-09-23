import mlflow
from src.pipeline.pipeline_v3 import AnimePipelineV3

mlflow.set_tracking_uri("http://mlflow:5000")

mlflow.set_experiment("anime-recsys-pipeline-v3")

def main():
    pipeline = AnimePipelineV3(sample_size=1000)
    anime, ratings_train = pipeline.load_data()

    params = {
        "max_features": 1000,
        "ngram_range": (1,1),
        "min_df": 2,
        "use_type": True
    }

    sim_matrix = pipeline.train_model(anime, **params)
    score = pipeline.evaluate_and_log(anime, sim_matrix, params)

    print(f"Pipeline V3 完成 ✅ Precision@10 = {score:.4f}")

if __name__ == "__main__":
    main()