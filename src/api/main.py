import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ValidationError
import pandas as pd
import os
import csv
from datetime import datetime

app = FastAPI(
    title="Anime Recommender API",
    description="FastAPI + MLflow 企業級推薦系統",
    version="2.3.0"
)

# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI is running 🚀"}

# === 輸入格式定義 ===
class RecommendRequest(BaseModel):
    user_id: str
    anime_titles: list[str]

class ABEvent(BaseModel):
    user_id: str
    model_name: str
    model_version: int
    recommended_title: str
    clicked: bool
    timestamp: datetime = datetime.utcnow()

# === 設定 MLflow ===
mlflow.set_tracking_uri("http://mlflow:5000")
model_cache = {}  # 模型快取，避免重複載入

def get_model(model_name: str):
    """依照模型名稱載入模型，若不存在則回傳 404"""
    if model_name not in model_cache:
        model_uri = f"models:/{model_name}/Staging"
        print(f"📦 Loading {model_uri} ...")
        try:
            model_cache[model_name] = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in Registry.")
    return model_cache[model_name]

# === 推薦 API ===
@app.post("/recommend")
def recommend(request: RecommendRequest, model_name: str = Query("AnimeRecsysModel")):
    try:
        if not request.anime_titles:
            raise HTTPException(status_code=400, detail="anime_titles cannot be empty.")
        model = get_model(model_name)
        df = pd.DataFrame(request.anime_titles)
        result = model.predict(df)
        return {
            "model_name": model_name,
            "input": request.anime_titles,
            "recommendations": result[0]
        }
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
# === 時間分流邏輯 ===
def choose_model_by_time():
    """偶數秒 → baseline；奇數秒 → TF-IDF"""
    sec = datetime.utcnow().second
    return "AnimeRecsysModel" if sec % 2 == 0 else "AnimeRecsysTFIDF"

# === A/B 測試端點 ===
@app.post("/recommend_ab")
def recommend_ab(request: RecommendRequest):
    """根據時間自動分流"""
    if not request.anime_titles:
        raise HTTPException(status_code=400, detail="anime_titles cannot be empty.")
    
    model_name = choose_model_by_time()
    model = get_model(model_name)
    result = model.predict(pd.DataFrame(request.anime_titles))

    print(f"🧠 User={request.user_id} 使用模型: {model_name}")

    return {
        "endpoint": "/recommend_ab",
        "user_id": request.user_id,
        "model_name": model_name,
        "recommendations": result[0],
        "timestamp": datetime.utcnow().isoformat()
    }

# === AB Test 紀錄 API ===
@app.post("/log-ab-event")
def log_ab_event(event: ABEvent):
    LOG_DIR = "/usr/mlflow/workspace/logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "ab_events.csv")

    try:
        file_exists = os.path.isfile(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "user_id", "model_name", "model_version", "recommended_title", "clicked"])
            writer.writerow([
                event.timestamp.isoformat(),
                event.user_id,
                event.model_name,
                event.model_version,
                event.recommended_title,
                event.clicked
            ])
        return {"message": "Event logged successfully ✅", "event": event.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write log: {e}")