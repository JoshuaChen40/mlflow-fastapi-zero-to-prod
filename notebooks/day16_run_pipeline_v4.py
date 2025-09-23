import mlflow
from src.pipeline.pipeline_v4 import AnimePipelineV4

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("anime-recsys-pipeline-v4")

def main():
    pipeline = AnimePipelineV4(sample_size=500)
    anime = pipeline.load_data()

    params = {
        "max_features": 500,
        "ngram_range": (1,1),
        "min_df": 2,
        "use_type": True
    }

    sim_matrix, vectorizer = pipeline.train_model(anime, **params)
    sample_dict, global_dict = pipeline.explain_and_log(anime, vectorizer, params)

    print("Pipeline V4 完成 ✅")
    print("\n🎯 單一樣本前 10 特徵：")
    for k, v in sample_dict.items():
        print(f"{k}: {v:.4f}")

    print("\n🌍 全資料集前 10 特徵：")
    for k, v in global_dict.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
