# VibeVoice-ASR 繁體中文微調 Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** 建立完整的工具鏈，讓使用者能用 Common Voice zh-TW 資料集對 VibeVoice-ASR 進行 LoRA 微調，使模型能辨識台灣華語並輸出繁體中文。

**Architecture:** 分為四個獨立元件：(1) 資料準備腳本將 Common Voice zh-TW 轉為 VibeVoice 訓練格式；(2) 訓練啟動腳本封裝 GH200 最佳化的 LoRA 微調命令；(3) 後處理模組提供 OpenCC 繁體轉換作為安全網；(4) 評估腳本比較 baseline 與微調模型的 CER 和繁體輸出品質。所有腳本設計為離線可執行，使用者在本地準備後帶到實驗室的 GH200 訓練。

**Tech Stack:** Python 3.9+, PyTorch, HuggingFace Transformers/PEFT/Datasets, OpenCC, jiwer

**Design Doc:** [2026-04-05-zh-tw-asr-design.md](./2026-04-05-zh-tw-asr-design.md)

---

### Task 1: 資料準備腳本 (prepare_common_voice.py)

**Files:**
- Create: `scripts/prepare_common_voice.py`
- Create: `scripts/requirements-zh-tw.txt`

**Step 1: 建立依賴清單**

建立 `scripts/requirements-zh-tw.txt`：

```text
datasets>=2.14.0
soundfile
tqdm
opencc-python-reimplemented
jiwer
```

**Step 2: 實作資料準備腳本**

建立 `scripts/prepare_common_voice.py`，功能：

1. 從 Hugging Face 下載 Common Voice zh-TW（train/test split）
2. 過濾掉過短（<1 秒）或過長（>30 秒）的音訊
3. 將每筆資料轉換為 VibeVoice 訓練格式的 JSON + 音檔配對
4. 支援 `--max_samples` 參數限制樣本數（用於快速試跑）
5. 輸出到指定目錄

```python
"""
Common Voice zh-TW → VibeVoice ASR 訓練格式轉換腳本。

用法：
    # 完整下載與轉換
    python scripts/prepare_common_voice.py --output_dir data/zh-tw-train

    # 快速試跑（只取 5000 筆）
    python scripts/prepare_common_voice.py --output_dir data/zh-tw-train --max_samples 5000

    # 準備測試集
    python scripts/prepare_common_voice.py --output_dir data/zh-tw-test --split test --max_samples 500
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf


def get_audio_duration(audio_path: str) -> float:
    """取得音訊檔案的時長（秒）。"""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return 0.0


def convert_sample(sample, output_dir: Path, idx: int) -> dict | None:
    """
    將 Common Voice 的一筆資料轉換為 VibeVoice 訓練格式。

    回傳：
        成功時回傳轉換後的 dict，失敗時回傳 None。
    """
    sentence = sample.get("sentence", "").strip()
    if not sentence:
        return None

    # 取得音訊資訊
    audio_data = sample.get("audio", {})
    if not audio_data:
        return None

    audio_array = audio_data.get("array")
    sampling_rate = audio_data.get("sampling_rate", 48000)

    if audio_array is None or len(audio_array) == 0:
        return None

    duration = len(audio_array) / sampling_rate

    # 過濾過短或過長的音訊
    if duration < 1.0 or duration > 30.0:
        return None

    # 儲存音訊為 WAV 檔
    audio_filename = f"{idx:06d}.wav"
    audio_path = output_dir / audio_filename

    sf.write(str(audio_path), audio_array, sampling_rate)

    # 建立 VibeVoice 格式的 JSON
    vibevoice_data = {
        "audio_duration": round(duration, 2),
        "audio_path": audio_filename,
        "segments": [
            {
                "speaker": 0,
                "text": sentence,
                "start": 0.0,
                "end": round(duration, 2),
            }
        ],
    }

    # 儲存 JSON
    json_filename = f"{idx:06d}.json"
    json_path = output_dir / json_filename

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)

    return vibevoice_data


def main():
    parser = argparse.ArgumentParser(
        description="將 Common Voice zh-TW 資料集轉換為 VibeVoice ASR 訓練格式"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="輸出目錄路徑",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="資料集 split (預設: train)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大樣本數（None = 全部）",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="17.0",
        help="Common Voice 版本 (預設: 17.0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 下載資料集
    dataset_name = f"mozilla-foundation/common_voice_{args.dataset_version.replace('.', '_')}"
    print(f"正在從 Hugging Face 載入 {dataset_name} (zh-TW, {args.split})...")
    print("（首次下載可能需要一些時間）")

    dataset = load_dataset(
        dataset_name,
        "zh-TW",
        split=args.split,
        trust_remote_code=True,
    )

    total = len(dataset)
    if args.max_samples and args.max_samples < total:
        # 隨機取樣以確保多樣性
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
        total = len(dataset)

    print(f"共 {total} 筆資料待處理")

    # 轉換
    converted = 0
    skipped = 0

    for i, sample in enumerate(tqdm(dataset, desc="轉換中")):
        result = convert_sample(sample, output_dir, converted)
        if result is not None:
            converted += 1
        else:
            skipped += 1

    # 輸出統計
    print(f"\n{'='*50}")
    print(f"轉換完成！")
    print(f"  成功轉換: {converted} 筆")
    print(f"  跳過: {skipped} 筆")
    print(f"  輸出目錄: {output_dir}")
    print(f"{'='*50}")

    # 儲存轉換統計
    stats = {
        "dataset": dataset_name,
        "split": args.split,
        "total_samples": total,
        "converted": converted,
        "skipped": skipped,
        "language": "zh-TW",
    }
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"統計資訊已儲存至: {stats_path}")


if __name__ == "__main__":
    main()
```

**Step 3: 驗證腳本可執行**

Run:
```bash
pip install datasets soundfile tqdm
python scripts/prepare_common_voice.py --output_dir /tmp/cv_test --split test --max_samples 10
```

Expected: 產生 `/tmp/cv_test/` 目錄，內含 `000000.wav` + `000000.json` 等檔案配對，JSON 的 `segments[0].text` 為繁體中文。

**Step 4: Commit**

```bash
git add scripts/prepare_common_voice.py scripts/requirements-zh-tw.txt
git commit -m "feat: 新增 Common Voice zh-TW 資料準備腳本"
```

---

### Task 2: 繁中後處理模組 (postprocess_zh_tw.py)

**Files:**
- Create: `scripts/postprocess_zh_tw.py`

**Step 1: 實作後處理模組**

建立 `scripts/postprocess_zh_tw.py`，功能：

1. OpenCC `s2twp` 簡繁轉換（含台灣用語修正）
2. 偵測文字中殘留的簡體字比例
3. 可獨立使用也可被其他腳本 import

```python
"""
繁體中文後處理模組。

提供 OpenCC 簡繁轉換和繁體品質檢測功能。

用法（獨立執行）：
    echo "这个软件很好用" | python scripts/postprocess_zh_tw.py
    python scripts/postprocess_zh_tw.py --text "今天的内存使用率很高"
    python scripts/postprocess_zh_tw.py --file output.txt
"""

import argparse
import re
import sys
import json
from typing import Optional

try:
    import opencc

    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False


class TraditionalChinesePostProcessor:
    """繁體中文後處理器。"""

    def __init__(self, config: str = "s2twp"):
        """
        初始化後處理器。

        Args:
            config: OpenCC 轉換設定
                - s2twp: 簡體 → 繁體（台灣用語 + 詞彙轉換）推薦
                - s2tw: 簡體 → 繁體（僅字形，不轉詞彙）
                - s2t: 簡體 → 繁體（基本轉換）
        """
        if not HAS_OPENCC:
            raise ImportError(
                "需要安裝 opencc-python-reimplemented：\n"
                "  pip install opencc-python-reimplemented"
            )
        self.converter = opencc.OpenCC(config)
        self.config = config

    def convert(self, text: str) -> str:
        """將文字轉換為繁體中文（台灣用語）。"""
        if not text:
            return text
        return self.converter.convert(text)

    def convert_transcription_json(self, json_text: str) -> str:
        """
        轉換 VibeVoice ASR JSON 格式的轉錄結果。

        只轉換 Content 欄位的文字，保留 JSON 結構不變。
        """
        try:
            data = json.loads(json_text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "Content" in item:
                        item["Content"] = self.convert(item["Content"])
            elif isinstance(data, dict) and "Content" in data:
                data["Content"] = self.convert(data["Content"])
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            # 非 JSON 格式，直接轉換整段文字
            return self.convert(json_text)

    @staticmethod
    def detect_simplified_ratio(text: str) -> float:
        """
        偵測文字中簡體字的近似比例。

        使用常見簡體字字集進行比對。回傳 0.0~1.0 的比例值。
        """
        # 常見簡體字（在繁體中不存在或不同形的）
        simplified_chars = set(
            "与专业丝丢两严丰临为丽举乌书买乱了予争事二亏云产亩亲亿仅从仑仓仪们价众优伙会伟传伤"
            "伦伪体余佣侠侣侥侦侧侨侩侪侬俣俩俪俭债倾偻偿傥傧储儿兑兖兰关兴兹养兽冁冈冲决况冻"
            "净凉凑凤凫凭凯击凿刍划刘则刚创删别刬刭刮制刹剀剂剐剑剥剧劝办务劢动励劲劳势勋勐勚匀匦"
            "区医华协单卖卢卤卧卫却卺厂厅历厉压厌厍厕厢厣厦厨厩厮县叁参叆叇双变叙叠只台叶号叹叽吁吃"
            "后向吓吕吗吨听启吴呐呒呕呖呗员呙呛呜咏咙咛咝咤哌响哑哒哓哔哕哗哙哜哝哟唛唝唠唡唢唣"
        )

        chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        if not chinese_chars:
            return 0.0

        simplified_count = sum(1 for c in chinese_chars if c in simplified_chars)
        return simplified_count / len(chinese_chars)


def main():
    parser = argparse.ArgumentParser(description="繁體中文後處理工具")
    parser.add_argument("--text", type=str, help="要轉換的文字")
    parser.add_argument("--file", type=str, help="要轉換的檔案路徑")
    parser.add_argument(
        "--config",
        type=str,
        default="s2twp",
        help="OpenCC 設定 (預設: s2twp)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="只偵測簡體字比例，不轉換",
    )
    args = parser.parse_args()

    # 讀取輸入
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("請提供 --text、--file 或 stdin 輸入")
        sys.exit(1)

    if args.detect_only:
        ratio = TraditionalChinesePostProcessor.detect_simplified_ratio(text)
        print(f"簡體字比例: {ratio:.2%}")
        return

    processor = TraditionalChinesePostProcessor(config=args.config)
    result = processor.convert(text)
    print(result)


if __name__ == "__main__":
    main()
```

**Step 2: 驗證後處理模組**

Run:
```bash
pip install opencc-python-reimplemented
python scripts/postprocess_zh_tw.py --text "这个软件的内存使用率很高"
```

Expected: 輸出 `這個軟體的記憶體使用率很高`

Run:
```bash
python scripts/postprocess_zh_tw.py --detect-only --text "这个软件很好"
```

Expected: 輸出簡體字比例 > 0%

**Step 3: Commit**

```bash
git add scripts/postprocess_zh_tw.py
git commit -m "feat: 新增繁體中文後處理模組（OpenCC s2twp）"
```

---

### Task 3: 訓練啟動腳本 (train_zh_tw.sh)

**Files:**
- Create: `scripts/train_zh_tw.sh`

**Step 1: 建立訓練啟動腳本**

```bash
#!/bin/bash
# VibeVoice-ASR 繁體中文 LoRA 微調啟動腳本
# 針對 NVIDIA GH200 (96GB HBM3) 最佳化
#
# 用法：
#   bash scripts/train_zh_tw.sh                    # 預設設定
#   bash scripts/train_zh_tw.sh --small             # 小量試跑 (~1hr)
#   bash scripts/train_zh_tw.sh --medium            # 中量訓練 (~4hr)
#   bash scripts/train_zh_tw.sh --full              # 全量訓練 (~12-24hr)
#
# 前置需求：
#   1. pip install -e .            (安裝 vibevoice)
#   2. pip install peft            (安裝 LoRA 支援)
#   3. 已執行 prepare_common_voice.py 準備資料

set -e

# 預設值
DATA_DIR="./data/zh-tw-train"
OUTPUT_DIR="./output/zh-tw-lora"
MODEL_PATH="microsoft/VibeVoice-ASR"
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LR="2e-4"
WARMUP_RATIO=0.05
LOGGING_STEPS=50
SAVE_STEPS=500

# 解析參數
PROFILE="medium"
for arg in "$@"; do
    case $arg in
        --small)
            PROFILE="small"
            NUM_EPOCHS=2
            SAVE_STEPS=200
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
    esac
done

echo "=================================================="
echo "VibeVoice-ASR 繁體中文 LoRA 微調"
echo "=================================================="
echo "設定檔: ${PROFILE}"
echo "資料目錄: ${DATA_DIR}"
echo "輸出目錄: ${OUTPUT_DIR}"
echo "模型: ${MODEL_PATH}"
echo "LoRA rank: ${LORA_R}, alpha: ${LORA_ALPHA}"
echo "Batch size: ${BATCH_SIZE} × ${GRAD_ACCUM} = 有效 batch $(( BATCH_SIZE * GRAD_ACCUM ))"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning rate: ${LR}"
echo "=================================================="

# 檢查資料目錄
if [ ! -d "${DATA_DIR}" ]; then
    echo "錯誤: 資料目錄不存在: ${DATA_DIR}"
    echo "請先執行: python scripts/prepare_common_voice.py --output_dir ${DATA_DIR}"
    exit 1
fi

# 計算資料筆數
NUM_SAMPLES=$(ls -1 "${DATA_DIR}"/*.json 2>/dev/null | wc -l)
echo "資料筆數: ${NUM_SAMPLES}"

if [ "${NUM_SAMPLES}" -eq 0 ]; then
    echo "錯誤: 資料目錄中沒有 JSON 檔案"
    exit 1
fi

# 建立輸出目錄
mkdir -p "${OUTPUT_DIR}"

# 啟動訓練
echo ""
echo "開始訓練..."
echo ""

torchrun --nproc_per_node=1 finetuning-asr/lora_finetune.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 3 \
    --gradient_checkpointing \
    --bf16 \
    --report_to none

echo ""
echo "=================================================="
echo "訓練完成！"
echo "LoRA 權重已儲存至: ${OUTPUT_DIR}"
echo ""
echo "下一步："
echo "  1. 推論測試: python scripts/inference_zh_tw.py --lora_path ${OUTPUT_DIR} --audio_file <音訊檔>"
echo "  2. 評估效果: python scripts/evaluate_zh_tw.py --lora_path ${OUTPUT_DIR} --test_dir data/zh-tw-test"
echo "  3. 合併模型: python scripts/merge_lora.py --lora_path ${OUTPUT_DIR} --output_path models/vibevoice-asr-zh-tw"
echo "=================================================="
```

**Step 2: 設定可執行權限並驗證語法**

Run:
```bash
chmod +x scripts/train_zh_tw.sh
bash -n scripts/train_zh_tw.sh
```

Expected: 無語法錯誤

**Step 3: Commit**

```bash
git add scripts/train_zh_tw.sh
git commit -m "feat: 新增 GH200 最佳化的繁中訓練啟動腳本"
```

---

### Task 4: 推論腳本 (inference_zh_tw.py)

**Files:**
- Create: `scripts/inference_zh_tw.py`

**Step 1: 實作帶後處理的推論腳本**

```python
"""
VibeVoice-ASR 繁體中文推論腳本。

整合 LoRA 微調模型 + OpenCC 後處理的完整推論管道。

用法：
    # 用 LoRA 推論
    python scripts/inference_zh_tw.py \
        --lora_path ./output/zh-tw-lora \
        --audio_file test.wav

    # 用合併後的模型推論
    python scripts/inference_zh_tw.py \
        --model_path ./models/vibevoice-asr-zh-tw \
        --audio_file test.wav

    # 停用後處理（純模型輸出）
    python scripts/inference_zh_tw.py \
        --lora_path ./output/zh-tw-lora \
        --audio_file test.wav \
        --no-postprocess
"""

import argparse
import json
import sys
import os
import time

import torch

# 新增根目錄到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peft import PeftModel
from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

try:
    from postprocess_zh_tw import TraditionalChinesePostProcessor

    HAS_POSTPROCESS = True
except ImportError:
    try:
        from scripts.postprocess_zh_tw import TraditionalChinesePostProcessor

        HAS_POSTPROCESS = True
    except ImportError:
        HAS_POSTPROCESS = False


def load_model(
    base_model_path: str,
    lora_path: str | None = None,
    device: str = "cuda",
    dtype=torch.bfloat16,
):
    """載入模型（支援 LoRA 和合併模型）。"""
    print(f"載入模型: {base_model_path}")

    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )

    attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map=device if device == "auto" else None,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    if device != "auto":
        model = model.to(device)

    if lora_path:
        print(f"載入 LoRA 權重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, processor


def transcribe(
    model,
    processor,
    audio_path: str,
    max_new_tokens: int = 8192,
    context_info: str | None = None,
    device: str = "cuda",
    use_postprocess: bool = True,
):
    """轉錄音訊並套用繁中後處理。"""
    print(f"轉錄: {audio_path}")

    inputs = processor(
        audio=audio_path,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )

    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": False,
    }

    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_config)

    generation_time = time.time() - start_time

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    # 後處理
    final_text = raw_text
    if use_postprocess and HAS_POSTPROCESS:
        pp = TraditionalChinesePostProcessor(config="s2twp")
        final_text = pp.convert_transcription_json(raw_text)

    # 解析結構化輸出
    try:
        segments = processor.post_process_transcription(final_text)
    except Exception:
        segments = []

    return {
        "raw_text": raw_text,
        "final_text": final_text,
        "segments": segments,
        "generation_time": generation_time,
        "postprocessed": use_postprocess and HAS_POSTPROCESS,
    }


def main():
    parser = argparse.ArgumentParser(description="VibeVoice-ASR 繁體中文推論")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="基礎模型路徑或合併後模型路徑",
    )
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 權重路徑")
    parser.add_argument(
        "--audio_file", type=str, required=True, help="音訊檔案路徑"
    )
    parser.add_argument(
        "--context_info", type=str, default=None, help="上下文資訊（熱詞等）"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=8192, help="最大生成 token 數"
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="停用 OpenCC 後處理",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32

    model, processor = load_model(
        base_model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )

    result = transcribe(
        model=model,
        processor=processor,
        audio_path=args.audio_file,
        max_new_tokens=args.max_new_tokens,
        context_info=args.context_info,
        device=args.device,
        use_postprocess=not args.no_postprocess,
    )

    print(f"\n{'='*60}")
    print(f"轉錄結果（生成時間: {result['generation_time']:.2f}s）")
    print(f"後處理: {'已套用' if result['postprocessed'] else '未套用'}")
    print(f"{'='*60}")

    if result["segments"]:
        for seg in result["segments"]:
            start = seg.get("start_time", "?")
            end = seg.get("end_time", "?")
            speaker = seg.get("speaker_id", "?")
            text = seg.get("text", "")
            print(f"[{start}-{end}] 說話者 {speaker}: {text}")
    else:
        print(result["final_text"])


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/inference_zh_tw.py
git commit -m "feat: 新增繁中推論腳本（整合 LoRA + OpenCC 後處理）"
```

---

### Task 5: 評估腳本 (evaluate_zh_tw.py)

**Files:**
- Create: `scripts/evaluate_zh_tw.py`

**Step 1: 實作評估腳本**

```python
"""
VibeVoice-ASR 繁體中文微調評估腳本。

比較 baseline 模型和微調模型在 Common Voice zh-TW test set 上的表現。

用法：
    # 評估 LoRA 微調模型
    python scripts/evaluate_zh_tw.py \
        --lora_path ./output/zh-tw-lora \
        --test_dir data/zh-tw-test \
        --max_samples 100

    # 評估合併後的模型
    python scripts/evaluate_zh_tw.py \
        --model_path ./models/vibevoice-asr-zh-tw \
        --test_dir data/zh-tw-test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from jiwer import cer as compute_cer
except ImportError:
    print("需要安裝 jiwer: pip install jiwer")
    sys.exit(1)

from scripts.inference_zh_tw import load_model, transcribe
from scripts.postprocess_zh_tw import TraditionalChinesePostProcessor


def load_test_data(test_dir: str, max_samples: int | None = None):
    """載入測試資料。"""
    test_path = Path(test_dir)
    samples = []

    for json_path in sorted(test_path.glob("*.json")):
        if json_path.name == "dataset_stats.json":
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        audio_path = test_path / data["audio_path"]
        if not audio_path.exists():
            continue

        reference_text = " ".join(
            seg["text"] for seg in data.get("segments", [])
        )

        samples.append(
            {
                "audio_path": str(audio_path),
                "reference": reference_text,
                "duration": data.get("audio_duration", 0),
            }
        )

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    return samples


def evaluate(
    model,
    processor,
    samples,
    device: str = "cuda",
    use_postprocess: bool = True,
):
    """執行評估並回傳指標。"""
    references = []
    hypotheses = []
    simplified_ratios = []
    total_time = 0

    pp = TraditionalChinesePostProcessor(config="s2twp")

    for sample in tqdm(samples, desc="評估中"):
        try:
            result = transcribe(
                model=model,
                processor=processor,
                audio_path=sample["audio_path"],
                device=device,
                use_postprocess=use_postprocess,
            )

            # 從 segments 中提取文字
            if result["segments"]:
                hyp_text = " ".join(
                    seg.get("text", "") for seg in result["segments"]
                )
            else:
                hyp_text = result["final_text"]

            references.append(sample["reference"])
            hypotheses.append(hyp_text)
            total_time += result["generation_time"]

            # 偵測簡體殘留
            ratio = pp.detect_simplified_ratio(hyp_text)
            simplified_ratios.append(ratio)

        except Exception as e:
            print(f"跳過 {sample['audio_path']}: {e}")
            continue

    # 計算指標
    if not references:
        return {"error": "沒有成功評估的樣本"}

    overall_cer = compute_cer(references, hypotheses)
    avg_simplified = sum(simplified_ratios) / len(simplified_ratios)
    traditional_rate = sum(1 for r in simplified_ratios if r < 0.01) / len(
        simplified_ratios
    )

    return {
        "num_samples": len(references),
        "cer": round(overall_cer * 100, 2),
        "avg_simplified_ratio": round(avg_simplified * 100, 2),
        "traditional_rate": round(traditional_rate * 100, 2),
        "total_time_sec": round(total_time, 2),
        "avg_time_per_sample": round(total_time / len(references), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="VibeVoice-ASR 繁中評估")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-ASR",
    )
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--no-postprocess", action="store_true"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="評估報告 JSON 輸出路徑",
    )
    args = parser.parse_args()

    # 載入測試資料
    samples = load_test_data(args.test_dir, args.max_samples)
    print(f"載入 {len(samples)} 筆測試資料")

    if not samples:
        print("錯誤: 沒有可用的測試資料")
        sys.exit(1)

    # 載入模型
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model, processor = load_model(
        base_model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )

    # 評估
    results = evaluate(
        model=model,
        processor=processor,
        samples=samples,
        device=args.device,
        use_postprocess=not args.no_postprocess,
    )

    # 輸出結果
    print(f"\n{'='*60}")
    print("評估結果")
    print(f"{'='*60}")
    print(f"  樣本數: {results['num_samples']}")
    print(f"  CER: {results['cer']}%")
    print(f"  平均簡體字殘留: {results['avg_simplified_ratio']}%")
    print(f"  完全繁體比例: {results['traditional_rate']}%")
    print(f"  總用時: {results['total_time_sec']}s")
    print(f"  每筆平均: {results['avg_time_per_sample']}s")
    print(f"{'='*60}")

    # 儲存報告
    if args.output_report:
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"報告已儲存至: {args.output_report}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/evaluate_zh_tw.py
git commit -m "feat: 新增繁中微調評估腳本（CER + 繁體比例）"
```

---

### Task 6: LoRA 合併腳本 (merge_lora.py)

**Files:**
- Create: `scripts/merge_lora.py`

**Step 1: 實作合併腳本**

```python
"""
合併 LoRA 權重到基礎模型。

用法：
    python scripts/merge_lora.py \
        --base_model microsoft/VibeVoice-ASR \
        --lora_path ./output/zh-tw-lora \
        --output_path ./models/vibevoice-asr-zh-tw
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peft import PeftModel
from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def main():
    parser = argparse.ArgumentParser(description="合併 LoRA 權重到基礎模型")
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/VibeVoice-ASR",
    )
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32

    print(f"載入基礎模型: {args.base_model}")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=dtype,
        trust_remote_code=True,
    )

    print(f"載入 LoRA 權重: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("合併 LoRA 權重...")
    model = model.merge_and_unload()

    print(f"儲存合併後模型至: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)

    # 同時儲存 processor
    processor = VibeVoiceASRProcessor.from_pretrained(
        args.base_model,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    processor.save_pretrained(args.output_path)

    print("完成！")
    print(f"合併後模型已儲存至: {args.output_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/merge_lora.py
git commit -m "feat: 新增 LoRA 權重合併腳本"
```

---

### Task 7: 實驗室操作指南

**Files:**
- Create: `docs/zh-tw-finetuning-guide.md`

**Step 1: 撰寫完整的離線操作指南**

撰寫一份使用者可以列印或帶到實驗室的操作指南，包含：

1. 環境設定步驟
2. 資料準備確認清單
3. 訓練執行命令
4. 訓練監控方法
5. 結果帶回步驟
6. 常見問題排除

**Step 2: Commit**

```bash
git add docs/zh-tw-finetuning-guide.md
git commit -m "docs: 新增繁中微調實驗室操作指南"
```

---

Plan complete and saved to `docs/plans/2026-04-05-zh-tw-asr-finetuning.md`.

Next step: run `.agent/workflows/execute-plan.md` to execute this plan task-by-task in single-flow mode.
