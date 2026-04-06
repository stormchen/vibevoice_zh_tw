# VibeVoice 繁體中文微調模型 vLLM 部署指南 (GH200 伺服器)

這份文件說明如何在已經設定好 vLLM 的 GH200 伺服器上，部署你合併好的 VibeVoice 繁體中文微調模型。

## 環境假設
*   **模型路徑**：已合併好的模型位於 `/models/vibevoice-asr-zh-tw`
*   **硬體環境**：Nvidia GH200 伺服器
*   **程式碼層**：假設 GH200 上已 clone 了 `VibeVoice` 官方專案程式碼

我們提供兩種部署方式：**Docker 部署 (推薦)** 與 **Native Python 部署 (若已建制好全域/虛擬環境)**。

---

## 方式一：使用官方 vLLM Docker 部署 (推薦)

VibeVoice 官方推薦使用 vLLM 的 Docker 映像檔來避免音訊套件 (例如 FFmpeg) 與 vLLM 底層 C++ 依賴之間的衝突。

### 1. 執行 Docker 啟動指令

請在 GH200 伺服器上，進入 `VibeVoice` 的專案目錄後，執行以下指令。
注意我們將本機的模型目錄 `/models/vibevoice-asr-zh-tw` 掛載進 Container 內（路徑保持一致），並透過 `--model` 指定該路徑：

```bash
docker run -d --gpus all --name vibevoice-vllm-zh-tw \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -v /models/vibevoice-asr-zh-tw:/models/vibevoice-asr-zh-tw \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py --model /models/vibevoice-asr-zh-tw"
```

### 參數解說：
*   `-v $(pwd):/app`：將當前的 VibeVoice 原始碼掛載進去，提供啟動腳本與 Plugin。
*   `-v /models/vibevoice-asr-zh-tw:/models/vibevoice-asr-zh-tw`：將你已合併的模型掛載到 Container 內部。
*   `--model /models/vibevoice-asr-zh-tw`：指示 `start_server.py` 使用該本機模型，而非從 HuggingFace 下載預設模型。
*   **多卡調整 (Data Parallel)**：GH200 通常具備強大的運算與多顯卡配置，若需增加 Throughput (例如 2 卡 DP)，可在最後補上 `--dp 2`。

---

## 方式二：Native Python 環境部署

如果你已經在 GH200 上的系統或虛擬環境中安裝好 `vllm`，並希望直接在 Native 環境啟動，請依循以下步驟。

### 1. 安裝系統音訊依賴 (若尚未安裝)
因為 VibeVoice 需要處理音軌，你需要確保系統有 FFmpeg 和相關庫：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

### 2. 安裝 VibeVoice 專案的 vLLM 插件支援
進入 `VibeVoice` 目錄並安裝：

```bash
pip install -e .[vllm]
```

### 3. 使用啟動腳本執行
由於系統環境已建立，我們可以加上 `--skip-deps` 參數來略過腳本內的系統套件檢查，直接用指定模型啟動 vLLM：

```bash
python3 vllm_plugin/scripts/start_server.py \
  --model /models/vibevoice-asr-zh-tw \
  --port 8000 \
  --skip-deps
```

---

## 測試 API 功能

當模型成功 Load 到 GH200 GPU 並顯示啟動完成後，你可以透過呼叫 `/v1/chat/completions` OpenAI 相容端點來測試音訊辨識：

```bash
# 基本音訊辨識測試 (假設音訊檔放置於目前專案目錄下)
python3 vllm_plugin/tests/test_api.py test_audio.wav

# 若希望加強繁中特定領域專有名詞的辨識 (Hotwords / Prompt)
python3 vllm_plugin/tests/test_api.py test_audio.wav --hotwords "特定的繁體中文詞彙,例如品牌名稱"
```

## 效能與除錯建議
1. **GH200 記憶體優化**：若有 OOM 疑慮，可透過修改 `--gpu-memory-utilization` 參數 (如降至 0.8 或 0.9)。
2. **檢視 Logs**：若是透過 Docker 部署，可使用 `docker logs -f vibevoice-vllm-zh-tw` 即時監看推理狀態或錯誤訊息。
