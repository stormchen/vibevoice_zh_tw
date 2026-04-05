#!/usr/bin/env bash
# =============================================================================
# VibeVoice-ASR 繁體中文（台灣華語）LoRA 微調啟動腳本
# 針對 NVIDIA GH200 (96GB HBM3) 最佳化
# =============================================================================
#
# 用法：
#   bash scripts/train_zh_tw.sh                          # 中量訓練（預設）
#   bash scripts/train_zh_tw.sh --small                  # 小量試跑 (~1hr)
#   bash scripts/train_zh_tw.sh --medium                 # 中量訓練 (~4hr)
#   bash scripts/train_zh_tw.sh --full                   # 全量訓練 (~12-24hr)
#   bash scripts/train_zh_tw.sh --data_dir=<路徑>         # 自訂資料目錄
#   bash scripts/train_zh_tw.sh --output_dir=<路徑>       # 自訂輸出目錄
#
# 前置需求（請在實驗室 GH200 機器上執行）：
#   1. pip install -e .            # 安裝 VibeVoice 套件
#   2. pip install peft            # 安裝 LoRA 支援
#   3. pip install -r scripts/requirements-zh-tw.txt  # 安裝其他依賴
#   4. 執行資料準備：python scripts/prepare_common_voice.py --output_dir data/zh-tw-train
#
# 輸出：
#   - LoRA 權重：./output/zh-tw-lora/
#   - 訓練 log：標準輸出
# =============================================================================

set -euo pipefail

# ─── 預設參數 ─────────────────────────────────────────────────────────────────
DATA_DIR="./data/zh-tw-train"
OUTPUT_DIR="./output/zh-tw-lora"
MODEL_PATH="microsoft/VibeVoice-ASR"

# LoRA 超參數（針對 GH200 + 繁中微調最佳化）
LORA_R=32                  # 較高 rank，繁體學習需要更多表達力
LORA_ALPHA=64              # 2x rank（慣例）
LORA_DROPOUT=0.05

# 訓練超參數
NUM_EPOCHS=3
BATCH_SIZE=4               # GH200 記憶體充足，可用較大 batch
GRAD_ACCUM=4               # 有效 batch = BATCH_SIZE × GRAD_ACCUM = 16
LR="2e-4"                  # LoRA 微調典型學習率
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
LOGGING_STEPS=50
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3

# ─── 解析命令列參數 ──────────────────────────────────────────────────────────
PROFILE="medium"
for arg in "$@"; do
    case "${arg}" in
        --small)
            PROFILE="small"
            NUM_EPOCHS=2
            SAVE_STEPS=200
            LOGGING_STEPS=20
            ;;
        --medium)
            PROFILE="medium"
            ;;
        --full)
            PROFILE="full"
            NUM_EPOCHS=3
            SAVE_STEPS=1000
            ;;
        --data_dir=*)
            DATA_DIR="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        --lora_r=*)
            LORA_R="${arg#*=}"
            ;;
        --lr=*)
            LR="${arg#*=}"
            ;;
        --epochs=*)
            NUM_EPOCHS="${arg#*=}"
            ;;
        -h|--help)
            head -30 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "未知參數: ${arg}"
            echo "執行 bash $0 --help 查看說明"
            exit 1
            ;;
    esac
done

# ─── 印出設定摘要 ─────────────────────────────────────────────────────────────
echo "=================================================="
echo "VibeVoice-ASR 繁體中文 LoRA 微調"
echo "=================================================="
echo "  設定檔:        ${PROFILE}"
echo "  資料目錄:      ${DATA_DIR}"
echo "  輸出目錄:      ${OUTPUT_DIR}"
echo "  模型:          ${MODEL_PATH}"
echo "  LoRA rank:     ${LORA_R}  alpha: ${LORA_ALPHA}"
echo "  Batch:         ${BATCH_SIZE} × ${GRAD_ACCUM} = 有效 $(( BATCH_SIZE * GRAD_ACCUM ))"
echo "  Epochs:        ${NUM_EPOCHS}"
echo "  Learning rate: ${LR}"
echo "=================================================="

# ─── 環境檢查 ────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] 檢查環境..."

# 檢查 CUDA
if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" 2>/dev/null; then
    GPU_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')")
    echo "  ✅ GPU: ${GPU_INFO} (${GPU_MEM})"
else
    echo "  ⚠️  CUDA 不可用，將使用 CPU 訓練（非常慢，不建議）"
    BATCH_SIZE=1
fi

# 檢查 VibeVoice
if python3 -c "import vibevoice" 2>/dev/null; then
    echo "  ✅ VibeVoice 已安裝"
else
    echo "  ❌ VibeVoice 未安裝，請執行：pip install -e ."
    exit 1
fi

# 檢查 PEFT
if python3 -c "import peft" 2>/dev/null; then
    echo "  ✅ PEFT 已安裝"
else
    echo "  ❌ PEFT 未安裝，請執行：pip install peft"
    exit 1
fi

# ─── 資料目錄檢查 ────────────────────────────────────────────────────────────
echo ""
echo "[2/4] 檢查資料..."

if [ ! -d "${DATA_DIR}" ]; then
    echo "  ❌ 資料目錄不存在: ${DATA_DIR}"
    echo ""
    echo "  請先執行資料準備腳本："
    echo "    python scripts/prepare_common_voice.py --output_dir ${DATA_DIR}"
    exit 1
fi

# 計算 JSON 檔案數量（等於訓練樣本數）
NUM_SAMPLES=$(find "${DATA_DIR}" -name "*.json" ! -name "dataset_stats.json" | wc -l | tr -d ' ')

if [ "${NUM_SAMPLES}" -eq 0 ]; then
    echo "  ❌ 資料目錄中沒有 JSON 訓練檔案: ${DATA_DIR}"
    exit 1
fi

echo "  ✅ 訓練樣本數: ${NUM_SAMPLES}"

# 估算訓練時間（GH200 每步約 2-3 秒）
STEPS_PER_EPOCH=$(( NUM_SAMPLES / (BATCH_SIZE * GRAD_ACCUM) ))
TOTAL_STEPS=$(( STEPS_PER_EPOCH * NUM_EPOCHS ))
EST_MINUTES=$(( TOTAL_STEPS * 3 / 60 ))
echo "  📊 估計訓練步數: ${TOTAL_STEPS}（${NUM_EPOCHS} epochs）"
echo "  ⏱️  預估時間: ~${EST_MINUTES} 分鐘（基於 GH200 估算）"

# ─── 建立輸出目錄 ────────────────────────────────────────────────────────────
echo ""
echo "[3/4] 準備輸出目錄..."
mkdir -p "${OUTPUT_DIR}"
echo "  ✅ 輸出目錄: ${OUTPUT_DIR}"

# 儲存訓練設定以供日後參考
cat > "${OUTPUT_DIR}/training_config.json" << EOF
{
  "profile": "${PROFILE}",
  "model_path": "${MODEL_PATH}",
  "data_dir": "${DATA_DIR}",
  "num_samples": ${NUM_SAMPLES},
  "lora_r": ${LORA_R},
  "lora_alpha": ${LORA_ALPHA},
  "lora_dropout": ${LORA_DROPOUT},
  "num_train_epochs": ${NUM_EPOCHS},
  "per_device_train_batch_size": ${BATCH_SIZE},
  "gradient_accumulation_steps": ${GRAD_ACCUM},
  "effective_batch_size": $(( BATCH_SIZE * GRAD_ACCUM )),
  "learning_rate": "${LR}",
  "warmup_ratio": ${WARMUP_RATIO},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# ─── 啟動訓練 ────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] 開始訓練..."
echo ""
echo "  開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  按 Ctrl+C 可中斷（已儲存的 checkpoint 不會丟失）"
echo ""

# 記錄開始時間
START_TIME=$(date +%s)

torchrun --nproc_per_node=1 finetuning-asr/lora_finetune.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --gradient_checkpointing \
    --bf16 \
    --report_to none

# 計算實際耗時
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

# ─── 完成摘要 ────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "訓練完成！"
echo "  實際耗時:  ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  LoRA 權重:  ${OUTPUT_DIR}"
echo "=================================================="
echo ""
echo "下一步（擇一）："
echo ""
echo "  1. 快速測試推論："
echo "     python scripts/inference_zh_tw.py \\"
echo "         --lora_path ${OUTPUT_DIR} \\"
echo "         --audio_file <你的音訊檔>"
echo ""
echo "  2. 評估模型效果："
echo "     python scripts/evaluate_zh_tw.py \\"
echo "         --lora_path ${OUTPUT_DIR} \\"
echo "         --test_dir data/zh-tw-test"
echo ""
echo "  3. 合併 LoRA 到基礎模型（推論速度更快）："
echo "     python scripts/merge_lora.py \\"
echo "         --lora_path ${OUTPUT_DIR} \\"
echo "         --output_path models/vibevoice-asr-zh-tw"
echo ""
