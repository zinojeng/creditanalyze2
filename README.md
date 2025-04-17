# 糖尿病學會學分分析助手

## 專案簡介

「糖尿病學會學分分析助手」是一個使用 Streamlit 開發的網頁應用程式，用於自動化分析醫學會議文件並計算積分。系統能夠處理各種格式的文件（PDF、DOCX、圖片等），提取關鍵資訊，並進行智能評估，大幅度降低人工審核的時間與成本。

### 核心功能

- **文件解析**：支援 PDF、DOCX、JPG、JPEG、PNG 格式文件
- **內容提取**：使用 OCR 和 AI 技術從不同格式文件中提取文字
- **智能分析**：自動識別演講主題、講者、時間等資訊
- **積分計算**：根據會議內容和時間自動計算甲類或乙類積分
- **結果匯出**：將分析結果輸出為格式化的 Excel 檔案

## 存取資訊

- **GitHub 儲存庫**：https://github.com/zinojeng/creditanalyze2
- **部署地址**：[Zeabur 部署地址]

## 本地開發與執行

### 環境準備

1. 確保已安裝 Python 3.8 或更高版本
2. 克隆儲存庫：
   ```bash
   git clone https://github.com/zinojeng/creditanalyze2.git
   cd creditanalyze2
   ```

3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

4. 安裝系統依賴（適用於 Linux/macOS）：
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils

   # macOS
   brew install tesseract poppler
   ```

### 本地執行

1. 創建 `.env` 檔案，並添加 OpenAI API 金鑰：
   ```
   OPENAI_API_KEY=你的API金鑰
   ```

2. 啟動應用：
   ```bash
   streamlit run main.py
   ```

3. 瀏覽器會自動打開 `http://localhost:8501`

## 使用方法

1. 在側邊欄輸入 OpenAI API 金鑰
2. 選擇使用的 AI 模型（推薦使用 gpt-4.1-mini 或 gpt-4o）
3. 上傳需要分析的醫學會議文件
4. 等待系統處理，查看分析結果
5. 下載生成的 Excel 檔案

## 部署到 Zeabur

### 準備工作

1. 確保 GitHub 儲存庫中包含以下檔案：
   - `requirements.txt`：依賴項清單
   - `zeabur.json`：Zeabur 配置檔案
   - `Procfile`：部署命令
   - `runtime.txt`：Python 版本

2. 確保這些檔案已推送到 GitHub：
   ```bash
   git add .
   git commit -m "Update deployment files"
   git push origin main
   ```

### 部署步驟

1. **登入 Zeabur**
   - 訪問 https://zeabur.com
   - 使用 GitHub 帳戶登入

2. **創建新項目**
   - 點擊 "Create Project"
   - 命名為 "Credit Analyze" 或其他名稱

3. **部署服務**
   - 點擊 "Deploy Service"
   - 選擇 "GitHub" 
   - 選擇 `creditanalyze2` 儲存庫
   - 選擇 `main` 分支

4. **配置環境變數**
   - 找到 "Environment Variables" 部分
   - 添加 `OPENAI_API_KEY` 變數並設定你的 API 金鑰

5. **完成部署**
   - 等待部署完成
   - 使用提供的域名訪問應用

## 更新與維護

1. **本地修改**
   ```bash
   git add .
   git commit -m "描述你的更改"
   git push origin main
   ```

2. **更新部署**
   - 如果啟用了自動部署，Zeabur 會自動更新
   - 否則，需要在 Zeabur 控制台手動重新部署

## 常見問題排解

### PDF 無法正確解析
- 確保 PDF 不是掃描件，或使用清晰的掃描件
- 檢查是否安裝了 poppler-utils
- 嘗試使用不同的 OCR 引擎

### API 調用失敗
- 檢查 API 金鑰是否正確
- 確認 API 金鑰餘額是否充足
- 檢查網絡連接狀態

### 部署失敗
- 查看 Zeabur 日誌了解具體錯誤
- 確保所有系統依賴已在部署過程中安裝
- 檢查環境變數是否正確設置

## 版本資訊

當前版本：1.0.3

## 聯絡資訊

- **程式設計**: Tseng Yao Hsien
- **機構**: Tung's Taichung Metroharbor Hospital
- **聯絡**: LINE: zinojeng 