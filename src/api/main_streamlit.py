# 🎨 Streamlit 主應用入口（/src/api/main_streamlit.py）

import os
import pandas as pd
import streamlit as st
import requests
from datetime import datetime

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
ANIME_CSV_PATH = "/src/api/notebooks/data/anime_clean.csv"

st.set_page_config(page_title="🎬 Anime Recommender", layout="wide")
st.title("🎬 Anime Recommendation System")

st.markdown("""
這是一個基於 **MLflow + FastAPI** 的推薦系統展示 🎯  
請輸入暱稱並選擇你喜歡的動畫，我們會為你推薦相似作品！
""")

# --- Step 1. 使用者暱稱 ---
nickname = st.text_input("請輸入你的暱稱 👤", placeholder="例如：Josh、小智、NekoFan")
if not nickname:
    st.info("請輸入暱稱後再繼續。")

# --- Step 2. 載入動畫清單 ---
@st.cache_data
def load_anime_list():
    if not os.path.exists(ANIME_CSV_PATH):
        st.error(f"❌ 找不到資料檔案：{ANIME_CSV_PATH}")
        return []
    df = pd.read_csv(ANIME_CSV_PATH)
    return df["name"].dropna().unique().tolist()

anime_list = load_anime_list()

# --- Step 3. 使用者選擇動畫 ---
selected_anime = st.multiselect(
    "選擇你喜歡的動畫（最多5部） 🎥",
    anime_list,
    max_selections=5,
    placeholder="例如：Naruto、Bleach、Attack on Titan..."
)

# --- Step 4. 定義兩個輔助函式 ---
def get_recommendations(user_id: str, anime_titles: list[str]):
    """呼叫 FastAPI /recommend API 取得推薦清單"""
    payload = {"user_id": user_id, "anime_titles": anime_titles}
    params = {"model_name": "AnimeRecsysModel"}
    res = requests.post(f"{FASTAPI_URL}/recommend", json=payload, params=params)
    if res.status_code == 200:
        return res.json()
    else:
        st.error(f"❌ 無法取得推薦結果：{res.text}")
        return None


def log_click_event(user_id: str, model_name: str, model_version: int, title: str):
    """呼叫 FastAPI /log-ab-event 紀錄使用者行為"""
    event = {
        "user_id": user_id,
        "model_name": model_name,
        "model_version": model_version,
        "recommended_title": title,
        "clicked": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        r = requests.post(f"{FASTAPI_URL}/log-ab-event", json=event)
        if r.status_code == 200:
            st.toast(f"✅ 已紀錄點擊：{title}")
        else:
            st.warning(f"⚠️ 紀錄失敗：{r.text}")
    except requests.exceptions.RequestException:
        st.warning("⚠️ 無法連線至 FastAPI。")


# --- Step 5. 取得推薦結果 ---
if st.button("🚀 取得推薦結果"):
    if not nickname:
        st.warning("請先輸入暱稱。")
    elif not selected_anime:
        st.warning("請至少選擇一部動畫。")
    else:
        data = get_recommendations(nickname, selected_anime)
        if data:
            st.session_state["recommendations"] = data.get("recommendations", [])
            st.session_state["model_name"] = data.get("model_name", "AnimeRecsysModel")
            st.session_state["model_version"] = 1  # 如需版本控制可動態讀取
            st.success("✅ 推薦結果已更新！")

# --- Step 6. 顯示推薦結果並提供點擊事件 ---
if "recommendations" in st.session_state:
    recs = st.session_state["recommendations"]
    model_name = st.session_state.get("model_name", "AnimeRecsysModel")
    model_version = st.session_state.get("model_version", 1)

    if recs:
        st.markdown("---")
        st.subheader(f"✨ 為 **{nickname}** 推薦的動畫：")
        for i, title in enumerate(recs[:10], 1):
            if st.button(f"{i}. {title}", key=f"rec_{i}"):
                log_click_event(nickname, model_name, model_version, title)
    else:
        st.warning("⚠️ 沒有可顯示的推薦結果。")
