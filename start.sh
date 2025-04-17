#!/bin/bash
# 啟動腳本

# 安裝依賴
pip install -r requirements.txt

# 啟動 Streamlit 應用
streamlit run main.py --server.port $PORT --server.address 0.0.0.0 