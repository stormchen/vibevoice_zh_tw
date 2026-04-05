# VibeVoice 繁體中文 ASR 微調 — 開發交接筆記

> **建立日期**：2026-04-06
> **對象硬體**：NVIDIA GH200 144G HBM3e (aarch64)
> **分支**：`feat/zh-tw-asr-finetuning`
> **遠端 Repo**：`https://github.com/stormchen/vibevoice_zh_tw`

---

## 一、專案目標

將 Microsoft 的 [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) 模型透過 **LoRA (Low-Rank Adaptation)** 微調，使其能夠準確辨識**台灣華語 (zh-TW)** 並輸出**繁體中文**文字。

---

## 二、專案架構與檔案清單

### 核心腳本 (`scripts/`)

| 檔案 | 用途 |
|------|------|
| [prepare_common_voice.py](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/prepare_common_voice.py) | 從 Mozilla Data Collective API 下載 Common Voice zh-TW 並轉換為 VibeVoice JSON + WAV 格式 |
| [train_zh_tw.sh](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/train_zh_tw.sh) | GH200 最佳化的 LoRA 微調啟動腳本（支援 `--small`/`--medium`/`--full` 三種規模） |
| [inference_zh_tw.py](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/inference_zh_tw.py) | 整合 LoRA 模型推論 + OpenCC 簡轉繁後處理 |
| [evaluate_zh_tw.py](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/evaluate_zh_tw.py) | 自動化評估腳本，比較 Baseline 與微調模型的 CER 和繁體率 |
| [merge_lora.py](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/merge_lora.py) | 將 LoRA 權重合併到基礎模型，產出獨立可部署的模型 |
| [postprocess_zh_tw.py](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/postprocess_zh_tw.py) | OpenCC (s2twp) 簡轉繁模組 |
| [requirements-zh-tw.txt](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/requirements-zh-tw.txt) | 繁中微調專用的額外依賴套件 |

### 訓練核心 (`finetuning-asr/`)

| 檔案 | 用途 |
|------|------|
| [lora_finetune.py](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/finetuning-asr/lora_finetune.py) | LoRA 微調的主程式，含 Dataset、DataCollator、Trainer 邏輯 |

### 文件 (`docs/`)

| 檔案 | 用途 |
|------|------|
| [zh-tw-finetuning-guide.md](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/docs/zh-tw-finetuning-guide.md) | 原始的實驗室操作指南（5 階段流程） |

---

## 三、Git 提交歷程

以下為完整的開發軌跡（由舊到新），記錄了每一步的決策與修正：

```
5fccb0f feat: 新增 Common Voice zh-TW 資料準備腳本
7f95b08 feat: 新增繁體中文後處理模組（OpenCC s2twp）
523ca84 feat: 新增 GH200 最佳化的繁中訓練啟動腳本
660adba feat: 新增繁中推論腳本（整合 LoRA + OpenCC 後處理）
beac044 feat: 新增繁中微調評估腳本（CER + 繁體比例比較）
9b76599 feat: 新增 LoRA 權重合併腳本
c621775 docs: 新增繁中微調實驗室操作指南
6bf3d87 feat: add support for processing Common Voice datasets
5369370 docs: 更新指南以反映 MDC API 下載，並最佳化依賴
c0ac617 fix: tar.gz 解壓縮路徑問題，支援動態偵測 clips 目錄
896f8af perf: 改用順序讀取 tar 代替隨機存取，解決解壓縮過慢的問題
fb5b4f0 fix: 將 flash_attention_2 改為 sdpa 以適應實驗室 CUDA 環境
a69e8fb fix: 不使用 torchrun，純單卡訓練以避開 NVML/NCCL 初始化衝突
2388ff2 chore: 抑制預期內的 ffmpeg loader 備援警告
862af37 perf: 將 GH200 訓練規格調升為 16x1
```

---

## 四、實際操作流程（已驗證）

### 階段 1：環境設定

```bash
# 1. Clone 並切換到正確分支
git clone https://github.com/stormchen/vibevoice_zh_tw.git
cd vibevoice_zh_tw
git checkout feat/zh-tw-asr-finetuning

# 2. 建立 Python 虛擬環境
python3 -m venv venv
source venv/bin/activate

# 3. 安裝 VibeVoice 套件
pip install -e .

# 4. 安裝 LoRA 支援與繁中依賴
pip install peft
pip install -r scripts/requirements-zh-tw.txt

# 5. ⚠️ 重要：安裝與驅動版本匹配的 PyTorch
#    先確認驅動版本：nvidia-smi 輸出的 CUDA Version
#    若為 12.4（550 驅動），請執行：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 階段 2：資料準備

```bash
# 下載 Common Voice zh-TW 資料集（約 3GB）
# 需要 Mozilla Data Collective API Key
export MDC_API_KEY="你的_API_KEY"

# 全量下載（約 7394 筆訓練、約 5000+ 筆測試）
python3 scripts/prepare_common_voice.py \
    --api_key $MDC_API_KEY \
    --output_dir data/zh-tw-train \
    --split train

# 準備測試集
python3 scripts/prepare_common_voice.py \
    --tarball common_voice_zh_tw.tar.gz \
    --output_dir data/zh-tw-test \
    --split test \
    --max_samples 200
```

> [!TIP]
> 若已有下載好的 `common_voice_zh_tw.tar.gz`，可直接使用 `--tarball` 參數跳過下載步驟。

### 階段 3：模型微調

```bash
# 建議使用 tmux 避免 SSH 斷線導致訓練丟失
tmux new -s vibevoice-train

# 啟動全量訓練（GH200 約 20-70 分鐘）
bash scripts/train_zh_tw.sh \
    --full \
    --data_dir=data/zh-tw-train

# 脫離 tmux（訓練不中斷）：Ctrl+b 然後按 d
# 重新連回：tmux attach -t vibevoice-train
```

### 階段 4：評估與推論

```bash
# 評估模型效果
python3 scripts/evaluate_zh_tw.py \
    --lora_path output/zh-tw-lora \
    --test_dir data/zh-tw-test \
    --limit 50

# 單檔推論測試
python3 scripts/inference_zh_tw.py \
    --lora_path output/zh-tw-lora \
    --audio_file "音訊檔.wav"
```

### 階段 5：模型匯出

```bash
# 合併 LoRA 權重為獨立模型
python3 scripts/merge_lora.py \
    --lora_path output/zh-tw-lora \
    --output_path models/vibevoice-asr-zh-tw
```

---

## 五、踩坑紀錄與解決方案

這是本次開發過程中遇到的所有問題，按時間順序整理：

### 坑 1：Common Voice 資料集下架

> [!WARNING]
> HuggingFace 上的 Common Voice zh-TW 資料集已經下架，無法再用 `datasets` 套件直接下載。

**解決方案**：改用 [Mozilla Data Collective (MDC) API](https://datacollective.mozillafoundation.org) 下載。需要先到官網註冊並取得 API Key。

---

### 坑 2：tar.gz 解壓路徑錯誤（0 筆成功）

**現象**：`轉換完成：0 筆成功，5000 筆跳過`

**原因**：寫死 `zh-TW/clips/xxx.mp3` 路徑，但某些版本的壓縮檔實際路徑可能是 `cv-corpus-25.0-2025-01-20/zh-TW/clips/xxx.mp3`。

**修正**：[commit c0ac617](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/prepare_common_voice.py) — 改為動態偵測壓縮檔內的基礎目錄。

---

### 坑 3：tar.gz 解壓速度極慢（卡在 0%）

**現象**：跑了 30 分鐘，進度仍顯示 `0/5000 (0.0%)`。

**原因**：Python `tarfile` 模組在 `.tar.gz`（gzip 壓縮）上使用 `tar.getmember()` 進行隨機存取時，每次都會**從頭重新解壓整個檔案**。5000 筆 × 3GB = 15TB 的冗餘解壓量。

**修正**：[commit 896f8af](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/prepare_common_voice.py) — 改用**順序讀取 (sequential read)**，先建立目標檔案的字典，然後只掃描一次 tar 檔即完成。

---

### 坑 4：HuggingFace Cache 權限錯誤

**現象**：`PermissionError: [Errno 13] Permission denied: '/home/storm/.cache/huggingface/hub/.locks/models--Qwen--Qwen2.5-7B'`

**原因**：之前有人用不同使用者身份（如 `sudo`）執行過下載，導致 `.locks` 資料夾權限錯誤。

**修正**：
```bash
sudo rm -rf /home/storm/.cache/huggingface/hub/.locks/models--Qwen--Qwen2.5-7B
sudo chown -R storm:storm /home/storm/.cache/huggingface
```

---

### 坑 5：Flash Attention 2 編譯失敗

**現象**：`RuntimeError: The detected CUDA version (12.0) mismatches the version that was used to compile PyTorch (13.0)`

**原因**：`flash-attn` 套件需要從原始碼編譯 C++ 擴展，但系統 CUDA 版本和 PyTorch 編譯版本不一致。

**修正**：[commit fb5b4f0](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/finetuning-asr/lora_finetune.py) — 將 `attn_implementation` 從 `"flash_attention_2"` 改為 `"sdpa"`（PyTorch 內建，無需額外編譯）。

---

### 坑 6：NVML/NCCL 初始化失敗

**現象**：`ncclSystemError: nvmlInit_v2() failed: Driver/library version mismatch`

**原因**：`torchrun` 會強制初始化 NCCL 分散式後端，即使只用單卡。而 NCCL 依賴 NVML，當 NVML 版本和實際驅動不一致時就會爆炸。

**修正**：[commit a69e8fb](file:///Users/storm/Projects/VibeVoice/.worktrees/feat/zh-tw-asr-finetuning/scripts/train_zh_tw.sh) — 將啟動方式從 `torchrun --nproc_per_node=1` 改為直接用 `python3` 執行。單卡訓練完全不需要分散式後端。

---

### 坑 7：NVIDIA 驅動 / 函式庫版本不對齊

**現象**：`nvidia-smi` 報錯 `Failed to initialize NVML: Driver/library version mismatch`。

**根因分析**：
| 組件 | 版本 |
|------|------|
| 記憶體中的核心模組 (Kernel Module) | `550.163.01` |
| 硬碟上的 NVML 函式庫 | `580.126` |
| DKMS 已編譯模組 | `550.163.01` |

系統在背景自動更新（`apt upgrade`）了函式庫到 580，但核心模組沒有重新編譯和載入。

**修正**：
```bash
# 方法 A：將函式庫降回 550（與核心模組對齊）
sudo apt-get install --reinstall nvidia-driver-550

# 方法 B：將核心模組升到 580（與函式庫對齊）
sudo apt install nvidia-driver-580

# 無論哪種，最後都需要重新開機
sudo reboot
```

---

### 坑 8：PyTorch 與 CUDA 驅動版本不相容

**現象**：`CUDA initialization: The NVIDIA driver on your system is too old (found version 12040)` 和 `ValueError: Your setup doesn't support bf16/gpu`。

**原因**：`pip install torch` 預設會裝最新版 PyTorch（編譯給 CUDA 12.6 或更高），但實驗室的 550 驅動只支援到 CUDA 12.4。

**修正**：指定安裝對應 CUDA 12.4 的 PyTorch 版本：
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> [!IMPORTANT]
> **這是最容易被忽略的一步！** 請務必在安裝完成後驗證：
> ```bash
> python3 -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}, 版本: {torch.version.cuda}')"
> # 預期輸出：CUDA 可用: True, 版本: 12.4
> ```

---

## 六、最終訓練結果（小量試跑）

| 指標 | 數值 |
|------|------|
| 訓練樣本數 | 5,000 |
| Epoch 數 | 2 |
| 總訓練步數 | 625 |
| 訓練耗時 | 29 分 23 秒 |
| 最終 Train Loss | **0.1521** |
| Trainable 參數比例 | 0.92% (80M / 8.7B) |
| 每秒吞吐量 | 5.67 samples/sec |
| GPU 記憶體用量 | ~42 GB / 147 GB |

### 全量訓練（進行中）

| 指標 | 數值 |
|------|------|
| 訓練樣本數 | 7,394 |
| Epoch 數 | 3 |
| Batch 配置 | 16 × 1 = 有效 16 |
| 總訓練步數 | 1,386 |
| 預估耗時 | ~20 分鐘 |
| 初始 Loss | 0.10 → 快速降至 0.06 |
| GPU 使用功率 | 494W / 900W |
| GPU 記憶體用量 | ~42 GB / 147 GB |

---

## 七、關鍵技術決策摘要

| 決策 | 選擇 | 理由 |
|------|------|------|
| 微調方法 | LoRA | 只訓練 0.92% 參數，保留多語言能力 |
| 注意力機制 | SDPA | 避免 Flash Attention 編譯狀況 |
| 啟動方式 | `python3` (非 torchrun) | 避免 NCCL 初始化失敗 |
| 資料來源 | Mozilla Data Collective API  | HuggingFace 上的 Common Voice 已下架 |
| 壓縮檔讀取 | 順序讀取 + 字典查表 | 避免 gzip 隨機存取的效能災難 |
| PyTorch 版本 | cu124 | 必須和 550 驅動 (CUDA 12.4) 對齊 |
| Batch Size | 16 × 1 | GH200 有 144GB，足以支撐大 Batch |
| 訓練精度 | bf16 | Hopper 架構原生支援 |

---

## 八、後續待辦事項

- [ ] 全量訓練完成後，執行 `evaluate_zh_tw.py` 確認 CER 與繁體率
- [ ] 若效果良好，使用 `merge_lora.py` 合併權重為獨立模型
- [ ] 將合併後的模型整合至 vLLM 服務進行線上部署
- [ ] 考慮在更大資料集（如加入 `validated.tsv`）上重新訓練以進一步提升效果
- [ ] 建議鎖定 `nvidia-driver` 版本避免自動更新：`sudo apt-mark hold nvidia-driver-550`
- [ ] MDC API Key 需定期更換，不應寫死在任何程式碼中
