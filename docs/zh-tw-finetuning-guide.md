# VibeVoice-ASR 繁體中文微調：實驗室操作指南

> **適用對象**：在 GH200 實驗室機器上執行微調的操作員
> **版本**：2026-04-05
> **相關設計文件**：`docs/plans/2026-04-05-zh-tw-asr-design.md`

---

## 事前確認清單（在家）

在前往實驗室之前，請確認以下項目已完成：

- [ ] 本機已 clone 或 pull 最新的 VibeVoice 程式碼
- [ ] 確認 branch 在 `feat/zh-tw-asr-finetuning`
- [ ] 帶著：USB 隨身碟 / 筆電（用來傳輸資料）
- [ ] 確認 `scripts/` 目錄下有以下所有腳本：
  - [ ] `prepare_common_voice.py`
  - [ ] `train_zh_tw.sh`
  - [ ] `inference_zh_tw.py`
  - [ ] `evaluate_zh_tw.py`
  - [ ] `merge_lora.py`
  - [ ] `postprocess_zh_tw.py`
  - [ ] `requirements-zh-tw.txt`

---

## 第一階段：環境設定（約 15 分鐘）

### 步驟 1：確認程式碼

```bash
# 在 GH200 上取得最新程式碼
git clone https://github.com/<your-repo>/VibeVoice.git
cd VibeVoice
git checkout feat/zh-tw-asr-finetuning

# 確認腳本存在
ls scripts/
```

### 步驟 2：設定 Python 環境

```bash
# 建立虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝 VibeVoice（含核心依賴）
pip install -e .

# 安裝 LoRA 微調支援
pip install peft

# 安裝繁中微調所需套件
pip install -r scripts/requirements-zh-tw.txt

# 驗證安裝
python3 -c "import vibevoice; import peft; import opencc; print('所有套件安裝成功 ✅')"
```

### 步驟 3：設定 Hugging Face Token

Common Voice 需要同意授權條款才能下載。

```bash
# 方法 1（推薦）：使用 CLI 登入
huggingface-cli login
# 輸入你的 HF token（在 https://huggingface.co/settings/tokens 取得）

# 方法 2：設定環境變數
export HF_TOKEN=hf_xxxxxxxxxx
```

### 步驟 4：驗證 GPU

```bash
python3 -c "
import torch
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'GPU 名稱: {torch.cuda.get_device_name(0)}')
print(f'GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"
```

**預期輸出（GH200）：**
```
CUDA 可用: True
GPU 名稱: NVIDIA GH200 480GB
GPU 記憶體: 94.4 GB  （或接近此值）
```

---

## 第二階段：資料準備（20-60 分鐘）

### 策略 A：小量試跑（建議先做，確認流程正確）

```bash
# 下載 5000 筆訓練資料（約 15-20 分鐘）
python3 scripts/prepare_common_voice.py \
    --output_dir data/zh-tw-train-small \
    --max_samples 5000

# 下載 200 筆測試資料
python3 scripts/prepare_common_voice.py \
    --output_dir data/zh-tw-test \
    --split test \
    --max_samples 200
```

### 策略 B：中量訓練（效果較好）

```bash
# 下載 20000 筆訓練資料（約 60 分鐘）
python3 scripts/prepare_common_voice.py \
    --output_dir data/zh-tw-train \
    --max_samples 20000

# 下載 500 筆測試資料
python3 scripts/prepare_common_voice.py \
    --output_dir data/zh-tw-test \
    --split test \
    --max_samples 500
```

### 驗證資料準備結果

```bash
# 確認檔案存在且格式正確
ls data/zh-tw-train-small/ | head -10
cat data/zh-tw-train-small/000000.json
```

**預期看到（000000.json 內容）：**
```json
{
  "audio_duration": 5.2,
  "audio_path": "000000.wav",
  "segments": [
    {
      "speaker": 0,
      "text": "這是一段繁體中文",
      "start": 0.0,
      "end": 5.2
    }
  ]
}
```

> ⚠️ **注意**：`text` 欄位應為繁體中文字。若看到簡體字，請確認下載的是 `zh-TW`（台灣繁體）而非 `zh-CN`。

---

## 第三階段：模型微調

### 試跑訓練（約 1 小時，用小量資料驗證）

```bash
# 使用 --small 設定和小量資料
bash scripts/train_zh_tw.sh \
    --small \
    --data_dir=data/zh-tw-train-small \
    --output_dir=output/zh-tw-lora-small
```

**訓練過程中你會看到：**
```
[4/4] 開始訓練...
  開始時間: 2026-04-05 10:30:00
  
{'loss': 2.34, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.89, 'learning_rate': 0.00018, 'epoch': 0.3}
...
```

> **重要**：Loss 應該持續下降。若 Loss 不下降或突然爆炸（大於 100），請停止訓練並回報。

### 完整訓練（在試跑正常後才執行）

```bash
bash scripts/train_zh_tw.sh \
    --medium \
    --data_dir=data/zh-tw-train \
    --output_dir=output/zh-tw-lora
```

### 在背景執行訓練（可離開電腦）

```bash
# 使用 nohup 讓訓練在背景執行
nohup bash scripts/train_zh_tw.sh \
    --medium \
    --data_dir=data/zh-tw-train \
    --output_dir=output/zh-tw-lora \
    > logs/training.log 2>&1 &

# 查看 PID
echo "Training PID: $!"

# 即時查看 log
tail -f logs/training.log

# 或用 screen
screen -S training
bash scripts/train_zh_tw.sh --medium
# Ctrl+A 再按 D 可切離 screen
# screen -r training 重新連回
```

---

## 第四階段：快速驗證（訓練完成後）

### 立即測試推論

使用 Common Voice 的一個音訊檔測試模型是否正常輸出繁體：

```bash
# 用 test set 的第一個音訊測試
python3 scripts/inference_zh_tw.py \
    --lora_path output/zh-tw-lora-small \
    --audio_file data/zh-tw-test/000000.wav

# 預期看到繁體輸出：
# [0.0 → 5.2] 說話者 0：
#   今天天氣很好（繁體字）
```

### 看看有沒有改善（與 baseline 比較）

```bash
# 評估 100 筆（約 20 分鐘）
python3 scripts/evaluate_zh_tw.py \
    --lora_path output/zh-tw-lora-small \
    --test_dir data/zh-tw-test \
    --max_samples 100 \
    --output_report reports/eval_small.json

# 簡單閱讀報告
cat reports/eval_small.json
```

**解讀評估結果：**

| 指標 | 越好 | 微調後預期改善幅度 |
|------|------|-----------------|
| CER | 越低 | 5-20% 相對改善 |
| 繁體率 | 越高 | 從 <10% 提升到 >80% |
| 簡體殘留率 | 越低 | 從 >50% 降到 <20% |

---

## 第五階段：帶回結果

### 需要帶回的檔案

```bash
# LoRA 權重（最重要，約 200-500MB）
ls output/zh-tw-lora/
# adapter_config.json
# adapter_model.safetensors  ← 主要權重檔
# training_config.json

# 評估報告（可選，文字檔很小）
ls reports/

# 訓練 log（可選）
ls logs/
```

### 傳輸方法

```bash
# 方法 1：用 scp 傳到本機
scp -r <gh200-user>@<gh200-ip>:~/VibeVoice/output/zh-tw-lora \
    ~/Projects/VibeVoice/output/

# 方法 2：打包後用 USB 帶走
tar czf zh-tw-lora.tar.gz output/zh-tw-lora/ reports/
# 複製到 USB 後帶走

# 方法 3：上傳到 HuggingFace Hub（需要帳號）
huggingface-cli repo create vibevoice-asr-zh-tw
huggingface-cli upload vibevoice-asr-zh-tw output/zh-tw-lora/
```

---

## 本機驗證（回家後）

```bash
# 確認 LoRA 已放到正確位置
ls output/zh-tw-lora/

# 本機推論測試（需要 GPU，即使只是 MPS）
python3 scripts/inference_zh_tw.py \
    --lora_path output/zh-tw-lora \
    --audio_file <任意繁中音訊> \
    --device mps  # Mac M 系列用 mps，無 GPU 用 cpu

# 合併成獨立模型（可選，方便後續部署）
python3 scripts/merge_lora.py \
    --lora_path output/zh-tw-lora \
    --output_path models/vibevoice-asr-zh-tw
```

---

## 常見問題排除

### Q1：下載資料時出現授權錯誤
```
AuthenticationError: You must be authenticated to access this dataset.
```
**解決**：執行 `huggingface-cli login` 並確認已在 HuggingFace 網站同意 Common Voice 授權條款。

---

### Q2：CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```
**解決**：GH200 有 96GB，通常不會發生。若發生，嘗試：
```bash
# 嘗試降低 batch size
bash scripts/train_zh_tw.sh --small
# 或手動指定更小的 batch
BATCH_SIZE=1 GRAD_ACCUM=8 bash scripts/train_zh_tw.sh
```

---

### Q3：訓練 Loss 不下降
Loss 超過 50 步後仍沒有下降的趨勢。

**可能原因**：
- 資料格式錯誤（text 欄位為空或格式不對）
- 學習率太高或太低

**除錯步驟**：
```bash
# 檢查資料格式
python3 -c "
import json, glob
files = glob.glob('data/zh-tw-train-small/*.json')[:5]
for f in files:
    d = json.load(open(f))
    print(f'{f}: {d[\"segments\"][0][\"text\"][:20]!r}')
"
```

---

### Q4：找不到 Flash Attention
```
ImportError: flash_attn is not installed.
```
**解決**：
```bash
pip install flash-attn --no-build-isolation
```
若無法安裝，腳本會自動降級使用 SDPA（效能略低但可運行）。

---

### Q5：訓練被中斷，想從 checkpoint 繼續

VibeVoice 訓練每 `--save_steps` 步儲存一次 checkpoint。

```bash
# 查看現有的 checkpoint
ls output/zh-tw-lora/
# checkpoint-500/
# checkpoint-1000/
# ...

# 從最新 checkpoint 繼續（HuggingFace Trainer 自動支援）
bash scripts/train_zh_tw.sh \
    --data_dir=data/zh-tw-train \
    --output_dir=output/zh-tw-lora  # 同樣的 output_dir，會自動找最新 checkpoint
```

---

## 聯絡與回報

訓練完成後，請記錄以下資訊並回報：

| 項目 | 值 |
|------|---|
| 訓練資料筆數 | _____ |
| 訓練時長 | _____ |
| Baseline CER | _____ % |
| 微調後 CER | _____ % |
| 繁體率（Baseline） | _____ % |
| 繁體率（微調後） | _____ % |
| 異常/問題 | _____ |
