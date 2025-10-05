from fastapi import FastAPI

app = FastAPI(
    title="Anime Recommender API",
    description="FastAPI + MLflow 企業級推薦系統",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI is running 🚀"}