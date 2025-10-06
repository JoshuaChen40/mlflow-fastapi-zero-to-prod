import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI(
    title="Anime Recommender API",
    description="FastAPI + MLflow 企業級推薦系統",
    version="2.0.0"
)

# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI is running 🚀"}

class RecommendRequest(BaseModel):
    anime_titles: list[str]

# === 載入模型 ===
mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_URI = "models:/AnimeRecsysModel/Staging"
print(f"📦 Loading model from {MODEL_URI} ...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("✅ Model loaded successfully!")

# === 推薦 API ===
@app.post("/recommend")
def recommend(request: RecommendRequest):
    try:
        df_input = pd.DataFrame(request.anime_titles)
        result = model.predict(df_input)
        return {
            "input": request.anime_titles,
            "recommendations": result[0]
        }
    except Exception as e:
        return {"error": str(e)}