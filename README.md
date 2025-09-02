
# 🎬 Anime Recommender System with MLflow + FastAPI + Streamlit

本專案示範如何從 0 開始打造一個 **模型實驗室**，並整合：

- **MLflow**：實驗追蹤與模型管理  
- **PostgreSQL**：模型 metadata backend  
- **FastAPI**：推薦 API (`/recommend`、`/log-ab-event`)  
- **Streamlit**：使用者介面 + AB 測試  
- **Docker Compose**：一鍵建立開發環境  

---

## 🚀 系統架構

```plaintext
資料 (Kaggle Anime DB)
        │
        ▼
[Data Pipeline] → 清理後資料
        │
        ▼
[Model Training] → MLflow Tracking → MLflow Registry
        │
        ├── FastAPI → 提供推薦 API
        └── Streamlit → 前端介面 + AB 測試
````

---

## 📂 資料夾結構

```plaintext
/usr/mlflow
│── src/                  # 主程式碼（唯一入口）
│   ├── data_pipeline/    # 資料處理
│   ├── models/           # 模型訓練/推論
│   ├── api/              # FastAPI
│   ├── ui/               # Streamlit
│   └── __init__.py
│
│── workspace/            # 所有「輸出結果」集中這裡
│   ├── data/             # 原始 + 清理後資料
│   │   ├── raw/
│   │   └── processed/
│   ├── mlruns/           # MLflow artifacts & runs
│   ├── reports/          # EDA 報告、分析結果
│   └── logs/             # 訓練/服務日誌
│
│── config/               # 配置檔
│   ├── params.yaml
│   ├── logging.yaml
│   └── mlflow.env
│
│── docker/               # Docker 相關檔案
│   ├── requirements-dev.txt     # python-dev 環境（完整，含 pandas / sklearn）
│   ├── requirements-api.txt     # FastAPI 環境
│   ├── requirements-ui.txt      # Streamlit 環境
│   └── Dockerfile.dev           # python-dev 專用 Dockerfile
│   └── Dockerfile.api           # FastAPI 專用 Dockerfile
│   └── Dockerfile.ui            # Streamlit 專用 Dockerfile
│
│── notebooks/            # Jupyter 實驗
│── tests/                # 測試程式
│── docker-compose.yml    # 容器編排
│── README.md             # 專案說明
```

---

## 🔧 使用方式

### 1. 啟動環境

```bash
docker-compose up -d
```

服務啟動後：

* MLflow UI → [http://localhost:5000](http://localhost:5000)
* PostgreSQL → `localhost:5432`

### 2. 進入 Python 開發容器

```bash
docker exec -it python-dev bash
```

安裝依賴（只需一次）：

```bash
pip install -r requirements.txt
```

### 3. 執行資料處理

```bash
python src/data_pipeline/download_kaggle_dataset.py
python src/data_pipeline/eda_and_cleaning.py
```

### 4. 訓練模型並記錄到 MLflow

```bash
python src/models/user_based_cf.py
```

結果會出現在 **MLflow UI**，並將 artifacts 存到 `workspace/mlruns/`。

---

## ✅ 測試

執行單元測試：

```bash
pytest tests/
```

---

## 📓 開發說明

* **程式碼** → `src/`
* **輸出結果** → `workspace/`
* **配置檔** → `config/`
* **實驗 Notebook** → `notebooks/`

## 建議開發流程：

1. 在 `notebooks/` 做 EDA / 原型開發
2. 確認流程後移植到 `src/`
3. 輸出統一存放在 `workspace/`
4. 使用 MLflow 管理所有實驗
