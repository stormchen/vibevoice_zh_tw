"""
LoRA 權重合併腳本 — 將微調後的 LoRA 適配器合併進基礎模型。

合併後的模型是獨立的完整模型，推論時不再需要同時載入基礎模型和 LoRA 權重，
速度稍快且部署更方便。

什麼時候合併？
  - LoRA 評估後效果滿意，想正式部署時
  - 要使用 vLLM 部署時（vLLM 對 PeftModel 的支援較有限）
  - 要分发給其他人使用時

用法：
    python scripts/merge_lora.py \\
        --lora_path ./output/zh-tw-lora \\
        --output_path ./models/vibevoice-asr-zh-tw

    # 自訂基礎模型路徑（例如本地快取）
    python scripts/merge_lora.py \\
        --base_model ./models/VibeVoice-ASR \\
        --lora_path ./output/zh-tw-lora \\
        --output_path ./models/vibevoice-asr-zh-tw

合併後使用：
    python scripts/inference_zh_tw.py \\
        --model_path ./models/vibevoice-asr-zh-tw \\
        --audio_file test.wav
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


def verify_lora_path(lora_path: str) -> bool:
    """
    確認 LoRA 目錄包含必要的檔案。

    Args:
        lora_path: LoRA 權重目錄路徑。

    Returns:
        True 表示目錄有效。
    """
    path = Path(lora_path)
    if not path.exists():
        print(f"錯誤：LoRA 目錄不存在：{lora_path}")
        return False

    # 至少需要有 adapter_config.json
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"錯誤：找不到 adapter_config.json，這可能不是有效的 LoRA 目錄")
        print(f"  路徑：{lora_path}")
        return False

    return True


def check_disk_space(output_path: str, required_gb: float = 20.0) -> None:
    """檢查磁碟空間是否足夠（合併後模型約 15-20GB）。"""
    import shutil
    parent = Path(output_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(parent).free / (1024 ** 3)
    if free_gb < required_gb:
        print(
            f"  ⚠️  磁碟剩餘空間不足：{free_gb:.1f} GB（建議至少 {required_gb:.0f} GB）"
        )
        print("  合併後的模型約需 15-20 GB 空間")
    else:
        print(f"  ✅ 磁碟剩餘空間：{free_gb:.1f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="合併 LoRA 權重到 VibeVoice-ASR 基礎模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
注意事項：
  - 合併後的模型約需 15-20 GB 磁碟空間
  - 合併過程需要足夠的 RAM/VRAM（建議 >= 32GB）
  - 合併後的模型「不再需要」LoRA 路徑即可直接推論

範例：
  python scripts/merge_lora.py \\
      --lora_path ./output/zh-tw-lora \\
      --output_path ./models/vibevoice-asr-zh-tw

  # 合併後立即測試
  python scripts/inference_zh_tw.py \\
      --model_path ./models/vibevoice-asr-zh-tw \\
      --audio_file test.wav
""",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="基礎模型路徑（HuggingFace ID 或本地目錄，預設: microsoft/VibeVoice-ASR）",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="LoRA 權重目錄路徑（訓練完成的 output 目錄）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="合併後模型的儲存路徑",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="執行裝置（預設: cuda 若可用，否則 cpu）",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="跳過合併後的驗證步驟（加快速度）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VibeVoice-ASR LoRA 權重合併")
    print("=" * 60)
    print(f"  基礎模型：{args.base_model}")
    print(f"  LoRA 路徑：{args.lora_path}")
    print(f"  輸出路徑：{args.output_path}")
    print(f"  裝置：{args.device}")

    # ─── 前置檢查 ──────────────────────────────────────────────────────────
    print("\n[1/4] 前置檢查...")

    if not verify_lora_path(args.lora_path):
        sys.exit(1)
    print(f"  ✅ LoRA 目錄有效：{args.lora_path}")

    check_disk_space(args.output_path)

    # 印出 LoRA 設定
    adapter_config_path = Path(args.lora_path) / "adapter_config.json"
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    print(f"  LoRA rank: {adapter_config.get('r', '?')}")
    print(f"  LoRA alpha: {adapter_config.get('lora_alpha', '?')}")

    # ─── 載入模型 ──────────────────────────────────────────────────────────
    print("\n[2/4] 載入基礎模型和 LoRA 權重...")
    start_time = time.time()
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32

    from peft import PeftModel
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    print(f"  載入基礎模型：{args.base_model}")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=dtype,
        trust_remote_code=True,
    )

    if args.device != "auto":
        model = model.to(args.device)

    print(f"  載入 LoRA 權重：{args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    load_time = time.time() - start_time
    print(f"  ✅ 載入完成（耗時 {load_time:.1f}s）")

    # ─── 合併 ──────────────────────────────────────────────────────────────
    print("\n[3/4] 合併 LoRA 權重...")
    merge_start = time.time()
    model = model.merge_and_unload()
    merge_time = time.time() - merge_start
    print(f"  ✅ 合併完成（耗時 {merge_time:.1f}s）")

    # ─── 儲存 ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] 儲存合併後的模型...")
    save_start = time.time()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))

    # 同時儲存 Processor（推論時需要）
    print("  儲存 Processor...")
    processor = VibeVoiceASRProcessor.from_pretrained(
        args.base_model,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    processor.save_pretrained(str(output_path))

    # 儲存合併元資料（記錄來源以便追蹤）
    metadata = {
        "merged_from": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "lora_r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
        },
        "merge_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": "zh-TW",
        "description": "VibeVoice-ASR 繁體中文（台灣華語）微調模型",
    }
    with open(output_path / "merge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    save_time = time.time() - save_start
    total_time = time.time() - start_time

    # 顯示儲存的檔案
    saved_files = [f.name for f in output_path.iterdir() if f.is_file()]

    print(f"  ✅ 儲存完成（耗時 {save_time:.1f}s）")
    print(f"  儲存的檔案：{', '.join(saved_files[:5])}" + (f" 等共 {len(saved_files)} 個" if len(saved_files) > 5 else ""))

    print(f"\n{'='*60}")
    print(f"合併完成！總耗時 {total_time:.1f}s")
    print(f"  模型路徑：{args.output_path}")
    print(f"{'='*60}")
    print(f"\n下一步：")
    print(f"  # 測試推論")
    print(f"  python scripts/inference_zh_tw.py \\")
    print(f"      --model_path {args.output_path} \\")
    print(f"      --audio_file <音訊檔案>")
    print(f"\n  # 評估合併後模型")
    print(f"  python scripts/evaluate_zh_tw.py \\")
    print(f"      --model_path {args.output_path} \\")
    print(f"      --test_dir data/zh-tw-test \\")
    print(f"      --skip_baseline")


if __name__ == "__main__":
    main()
