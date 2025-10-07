import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import csv
from datetime import datetime

app = FastAPI(
    title="Anime Recommender API",
    description="FastAPI + MLflow 企業級推薦系統",
    version="2.0.0"
)

# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI is running 🚀"}

# === 輸入格式定義 ===
class RecommendRequest(BaseModel):
    anime_titles: list[str]

class ABEvent(BaseModel):
    user_id: str
    model_version: int
    recommended_title: str
    clicked: bool
    timestamp: datetime = datetime.utcnow()

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

# === 記錄 API (/log-ab-event) ===
@app.post("/log-ab-event")
def log_ab_event(event: ABEvent):
    LOG_DIR = "/usr/mlflow/workspace/logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "ab_events.csv")

    # 檢查檔案是否存在，若不存在就加上標題列
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "user_id", "model_version", "recommended_title", "clicked"])
        writer.writerow([
            event.timestamp.isoformat(),
            event.user_id,
            event.model_version,
            event.recommended_title,
            event.clicked
        ])
    return {"message": "Event logged successfully", "event": event.dict()}