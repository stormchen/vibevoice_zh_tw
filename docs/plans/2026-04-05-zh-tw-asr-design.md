# VibeVoice-ASR 繁體中文（台灣華語）微調設計文件

## 目標

讓 VibeVoice-ASR (7B) 模型能：
1. 正確辨識台灣華語口音
2. 直接輸出繁體中文轉錄（而非簡體）
3. 使用台灣在地詞彙（例如「軟體」而非「軟件」）

## 背景

### 現狀問題
- VibeVoice-ASR 的中文訓練資料（AISHELL-4、AliMeeting）全為簡體中文
- 即使辨識台灣華語音訊，模型也會輸出簡體字
- 台灣特有詞彙和用語無法正確處理

### 技術基礎
- 底層 LLM：Qwen2.5-7B（原生支援繁簡中文 token）
- 已有 LoRA 微調工具鏈（`finetuning-asr/lora_finetune.py`）
- 音訊編碼器（acoustic + semantic tokenizer）不需修改

## 設計方案：LoRA 微調 + 後處理

### 1. 資料管道

**資料來源**：Mozilla Common Voice zh-TW
- 開放授權（CC-0）
- 數百小時台灣華語母語者錄音
- 每筆有音檔 + 繁體中文文字稿
- 透過 Hugging Face 下載：`mozilla-foundation/common_voice_17_0`, `language=zh-TW`

**格式轉換**：
```
Common Voice 格式                VibeVoice 訓練格式
┌────────────────────┐          ┌──────────────────────────────┐
│ audio: MP3         │    →     │ audio_path: "cv_0001.mp3"    │
│ sentence: "繁體文"  │          │ segments: [{                 │
│ duration: 5.2s     │          │   speaker: 0,                │
└────────────────────┘          │   text: "繁體文",             │
                                │   start: 0.0,                │
                                │   end: 5.2                   │
                                │ }]                           │
                                └──────────────────────────────┘
```

**注意事項**：
- Common Voice 為短句（3-15 秒），與 VibeVoice 長音訊特性不同
- 目標是讓模型學會繁體輸出和台灣口音，長音訊能力由原模型保留
- 單一說話者（speaker: 0）

### 2. 訓練設定

**硬體**：NVIDIA GH200 (96GB HBM3)

**LoRA 設定**：
| 參數 | 值 | 說明 |
|------|---|------|
| lora_r | 32 | 較高 rank，學習繁簡映射需要更多表達力 |
| lora_alpha | 64 | 2x rank 慣例 |
| lora_dropout | 0.05 | 防止過擬合 |
| target_modules | q/k/v/o_proj + gate/up/down_proj | 注意力層 + MLP |

**訓練超參數**：
| 參數 | 值 | 說明 |
|------|---|------|
| batch_size | 4 | GH200 記憶體充足 |
| gradient_accumulation | 4 | 有效 batch = 16 |
| learning_rate | 2e-4 | LoRA 微調典型值 |
| epochs | 3 | 充分學習但不過擬合 |
| warmup_ratio | 0.05 | 穩定訓練初期 |
| bf16 | ✅ | 半精度加速 |
| gradient_checkpointing | ✅ | 節省記憶體 |

**訓練策略**：
1. 先用小量資料（5hr, ~5000 筆）試跑 → ~1 小時
2. 確認效果後用中量（20hr, ~20000 筆）→ ~4 小時
3. 需要更好效果再上全量

### 3. 後處理

即使微調後仍保留輕量後處理作為安全網：

```
ASR 輸出 → OpenCC s2twp 轉換 → 台灣用語修正 → 最終輸出
```

- **OpenCC 模式**：`s2twp`（Simplified → Traditional Taiwan + Phrases）
- 不只轉字形，也轉詞彙

### 4. 評估

**指標**：
- CER (Character Error Rate) — 中文 ASR 標準指標
- 繁體轉換準確率 — 輸出是否全為繁體
- 台灣用語準確率 — 抽樣檢查

**流程**：
1. Baseline（原始模型）→ 記錄 CER + 簡體比例
2. 微調模型 → 記錄 CER + 繁體比例
3. 微調 + 後處理 → 最終 CER
4. 比較三者

**資料**：Common Voice zh-TW test split

### 5. 部署

- LoRA 推論：`inference_lora.py`
- 合併推論：`merge_lora.py` → 獨立模型
- Gradio demo：整合繁中後處理

## 產出物清單

| 檔案 | 說明 |
|------|------|
| `scripts/prepare_common_voice.py` | 資料下載與格式轉換 |
| `scripts/train_zh_tw.sh` | 一鍵訓練啟動腳本 |
| `scripts/postprocess_zh_tw.py` | 繁中後處理模組 |
| `scripts/evaluate_zh_tw.py` | 評估腳本 |
| `scripts/merge_lora.py` | LoRA 權重合併 |
| `scripts/inference_zh_tw.py` | 帶後處理的推論 |

## 工作流程

```
本地準備（你的 Mac）          實驗室（GH200）           本地驗證
┌─────────────────┐       ┌──────────────────┐    ┌──────────────────┐
│ 1. 下載資料集    │       │ 4. 安裝環境       │    │ 7. 用 LoRA 推論   │
│ 2. 格式轉換      │  →    │ 5. 執行訓練       │ →  │ 8. 跑評估腳本     │
│ 3. 準備腳本      │  USB  │ 6. 帶回 LoRA 權重  │    │ 9. 比較效果       │
└─────────────────┘       └──────────────────┘    └──────────────────┘
```
