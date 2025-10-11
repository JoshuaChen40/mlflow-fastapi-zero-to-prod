# ⚖️ 雙模型推薦比較頁（/src/api/pages/ab_multiple.py）

import os
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
ANIME_CSV_PATH = "/src/api/notebooks/data/anime_clean.csv"

st.set_page_config(page_title="⚖️ 雙模型推薦比較", layout="wide")
st.title("⚖️ 雙模型推薦比較頁")

st.markdown("""
本頁同時顯示兩個模型的推薦結果，  
使用者可對比推薦清單並提供正負反饋，  
讓 CTR 更真實反映實際互動情況。
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

def get_recommendations(model_name, user_id, anime_titles):
    payload = {"user_id": user_id, "anime_titles": anime_titles}
    params = {"model_name": model_name}
    res = requests.post(f"{FASTAPI_URL}/recommend", json=payload, params=params)
    if res.status_code == 200:
        return res.json()
    else:
        st.error(f"❌ {model_name} 取得推薦失敗：{res.text}")
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
        r = requests.post(f"{FASTAPI_URL}/log-ab-event", json=event)
        if r.status_code != 200:
            st.warning(f"⚠️ 記錄失敗：{r.text}")
    except requests.exceptions.RequestException:
        st.warning("⚠️ 無法連線至 FastAPI")

if st.button("🚀 取得雙模型推薦結果"):
    if not nickname:
        st.warning("請先輸入暱稱。")
    elif not selected_anime:
        st.warning("請至少選擇一部動畫。")
    else:
        model_a, model_b = "AnimeRecsysModel", "AnimeRecsysTFIDF"
        res_a = get_recommendations(model_a, nickname, selected_anime)
        res_b = get_recommendations(model_b, nickname, selected_anime)
        if res_a and res_b:
            st.session_state["rec_a"], st.session_state["rec_b"] = res_a["recommendations"], res_b["recommendations"]
            st.session_state["model_a"], st.session_state["model_b"] = model_a, model_b
            st.success("✅ 已取得兩模型推薦結果！")

if "rec_a" in st.session_state and "rec_b" in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🧠 模型 A：{st.session_state['model_a']}")
        for i, title in enumerate(st.session_state["rec_a"][:10], 1):
            if st.button(f"A{i}. {title}", key=f"a_{i}"):
                log_click_event(nickname, st.session_state["model_a"], 1, title, page="ab_multiple", clicked=True)
        if st.button("😐 我都不喜歡模型 A 的推薦", key="dislike_a"):
            log_click_event(nickname, st.session_state["model_a"], 1, None, page="ab_multiple", clicked=False)
            st.info("已記錄：使用者對模型 A 的推薦不感興趣。")

    with col2:
        st.subheader(f"🎯 模型 B：{st.session_state['model_b']}")
        for i, title in enumerate(st.session_state["rec_b"][:10], 1):
            if st.button(f"B{i}. {title}", key=f"b_{i}"):
                log_click_event(nickname, st.session_state["model_b"], 1, title, page="ab_multiple", clicked=True)
        if st.button("😐 我都不喜歡模型 B 的推薦", key="dislike_b"):
            log_click_event(nickname, st.session_state["model_b"], 1, None, page="ab_multiple", clicked=False)
            st.info("已記錄：使用者對模型 B 的推薦不感興趣。")
