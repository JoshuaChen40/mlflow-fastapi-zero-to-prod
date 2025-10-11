# 📊 A/B Test 結果分析頁（/src/api/pages/ab_report.py）

import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ✅ 正確路徑：本機 workspace/logs 對應容器 /src/api/workspace/logs
LOG_PATH = "/src/api/workspace/logs/ab_events.csv"

st.set_page_config(page_title="📊 AB Test 分析", layout="wide")
st.title("📊 A/B Test 結果分析")

st.markdown("""
此頁面會從 `ab_events.csv` 中讀取紀錄，  
分析兩個推薦模型的表現差異（點擊率、使用者數、事件數）。
""")

# --- Step 1. 讀取紀錄檔 ---
@st.cache_data(ttl=5.0)
def load_logs():
    if not os.path.exists(LOG_PATH):
        st.warning("⚠️ 找不到 ab_events.csv")
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

# 🔄 側邊欄：重新整理報表
st.sidebar.markdown("### 🔄 重新整理報表")
if st.sidebar.button("重新載入資料"):
    st.cache_data.clear()
    st.rerun()

df = load_logs()
st.info(f"📦 已載入 {len(df)} 筆事件紀錄。")
if df.empty:
    st.stop()

st.markdown("### 🧾 最新事件紀錄（最近 10 筆）")
st.dataframe(df.tail(10), use_container_width=True)

# --- Step 2. 統計每個模型的點擊狀況 ---
summary = (
    df.groupby("model_name")
    .agg(
        total_clicks=("clicked", "sum"),
        unique_users=("user_id", pd.Series.nunique),
        total_events=("user_id", "count"),
    )
    .reset_index()
)

summary["CTR(%)"] = round(summary["total_clicks"] / summary["total_events"] * 100, 2)

st.markdown("### 📈 模型表現摘要")
st.dataframe(summary, use_container_width=True)

# --- Step 3. 視覺化 ---
st.markdown("### 📊 點擊率比較圖")
fig = px.bar(
    summary,
    x="model_name",
    y="CTR(%)",
    color="model_name",
    text="CTR(%)",
    title="不同模型的點擊率比較"
)
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 👥 使用者參與數量")
fig2 = px.bar(
    summary,
    x="model_name",
    y="unique_users",
    color="model_name",
    text="unique_users",
    title="各模型參與使用者數"
)
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

st.success("✅ 分析完成！您可以透過此頁面觀察模型互動差異。")
