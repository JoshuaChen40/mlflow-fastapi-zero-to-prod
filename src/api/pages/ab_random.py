# 🎲 隨機分流推薦頁（/src/api/pages/ab_random.py）

import os
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
ANIME_CSV_PATH = "/src/api/notebooks/data/anime_clean.csv"

st.set_page_config(page_title="🎲 隨機分流推薦", layout="wide")
st.title("🎲 A/B Test 隨機分流頁")

st.markdown("""
本頁使用 FastAPI `/recommend_ab` 隨機分流至不同模型，  
並新增「我都不喜歡」按鈕記錄負樣本，使 CTR 統計更真實。
""")

nickname = st.text_input("請輸入你的暱稱 👤", placeholder="例如：Josh、Mina、Ken")
if not nickname:
    st.info("請輸入暱稱後再繼續。")

@st.cache_data
def load_anime_list():
    if not os.path.exists(ANIME_CSV_PATH):
        st.error(f"❌ 找不到資料檔案：{ANIME_CSV_PATH}")
        return []
    df = pd.read_csv(ANIME_CSV_PATH)
    return df["name"].dropna().unique().tolist()

anime_list = load_anime_list()
selected_anime = st.multiselect("選擇你喜歡的動畫（最多5部） 🎥", anime_list, max_selections=5)

def get_random_recommend(user_id, anime_titles):
    payload = {"user_id": user_id, "anime_titles": anime_titles}
    res = requests.post(f"{FASTAPI_URL}/recommend_ab", json=payload)
    if res.status_code == 200:
        return res.json()
    else:
        st.error(f"❌ recommend_ab 取得推薦失敗：{res.text}")
        return None

def log_click_event(user_id, model_name, model_version, title, page, clicked=True):
    event = {
        "user_id": user_id,
        "model_name": model_name,
        "model_version": model_version,
        "recommended_title": title if title else None,
        "clicked": clicked,
        "timestamp": datetime.utcnow().isoformat(),
        "page": page
    }
    try:
        requests.post(f"{FASTAPI_URL}/log-ab-event", json=event)
    except requests.exceptions.RequestException:
        st.warning("⚠️ 無法記錄事件")

if st.button("🚀 取得隨機推薦結果"):
    if not nickname:
        st.warning("請先輸入暱稱。")
    elif not selected_anime:
        st.warning("請至少選擇一部動畫。")
    else:
        res = get_random_recommend(nickname, selected_anime)
        if res:
            st.session_state["random_recs"] = res["recommendations"]
            st.session_state["model_name"] = res["model_name"]
            st.success(f"✅ 本次使用模型：{st.session_state['model_name']}")

if "random_recs" in st.session_state:
    model_name = st.session_state["model_name"]
    recs = st.session_state["random_recs"]

    st.markdown("---")
    st.subheader(f"✨ 模型：{model_name}")
    for i, title in enumerate(recs[:10], 1):
        if st.button(f"{i}. {title}", key=f"r_{i}"):
            log_click_event(nickname, model_name, 1, title, page="ab_random", clicked=True)
    if st.button("😐 我都不喜歡以上推薦"):
        log_click_event(nickname, model_name, 1, None, page="ab_random", clicked=False)
        st.info("已記錄：使用者對本輪推薦沒有興趣。")
