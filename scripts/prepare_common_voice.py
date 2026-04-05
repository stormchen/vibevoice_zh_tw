"""
Common Voice zh-TW → VibeVoice ASR 訓練格式轉換腳本。

用法：
    # 完整下載與轉換（train split，建議先用 --max_samples 試跑）
    python scripts/prepare_common_voice.py --output_dir data/zh-tw-train

    # 快速試跑（只取 5000 筆）
    python scripts/prepare_common_voice.py --output_dir data/zh-tw-train --max_samples 5000

    # 準備測試集
    python scripts/prepare_common_voice.py --output_dir data/zh-tw-test --split test --max_samples 500

前置需求：
    pip install -r scripts/requirements-zh-tw.txt
    # 需要設定 Hugging Face token（如 Common Voice 需要同意授權條款）：
    #   huggingface-cli login
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def get_audio_duration_from_array(audio_array, sampling_rate: int) -> float:
    """由音訊陣列計算時長（秒）。"""
    return len(audio_array) / sampling_rate


def convert_sample(sample: dict, output_dir: Path, idx: int) -> dict | None:
    """
    將 Common Voice 的一筆資料轉換為 VibeVoice 訓練格式。

    Args:
        sample: Common Voice 資料集的一筆樣本。
        output_dir: 輸出目錄。
        idx: 樣本索引（用於命名輸出檔案）。

    Returns:
        成功時回傳 VibeVoice 格式的 dict，失敗時回傳 None。
    """
    try:
        import soundfile as sf
    except ImportError:
        print("需要安裝 soundfile：pip install soundfile")
        sys.exit(1)

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

    duration = get_audio_duration_from_array(audio_array, sampling_rate)

    # 過濾過短或過長的音訊
    if duration < 1.0 or duration > 30.0:
        return None

    # 儲存音訊為 WAV 檔（VibeVoice 支援多種格式，WAV 最通用）
    audio_filename = f"{idx:06d}.wav"
    audio_path = output_dir / audio_filename

    sf.write(str(audio_path), audio_array, sampling_rate)

    # 建立 VibeVoice 格式的標籤 JSON
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
        # 可選：加入語言標記作為 context，幫助模型理解這是繁體中文
        "customized_context": ["繁體中文", "Traditional Chinese", "zh-TW"],
    }

    # 儲存 JSON 標籤
    json_filename = f"{idx:06d}.json"
    json_path = output_dir / json_filename

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)

    return vibevoice_data


def main():
    parser = argparse.ArgumentParser(
        description="將 Common Voice zh-TW 資料集轉換為 VibeVoice ASR 訓練格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 試跑（5000 筆，約 10-15 分鐘）
  python scripts/prepare_common_voice.py \\
      --output_dir data/zh-tw-train-small \\
      --max_samples 5000

  # 中量訓練資料（20000 筆）
  python scripts/prepare_common_voice.py \\
      --output_dir data/zh-tw-train \\
      --max_samples 20000

  # 評估用測試集
  python scripts/prepare_common_voice.py \\
      --output_dir data/zh-tw-test \\
      --split test \\
      --max_samples 500
""",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="輸出目錄路徑（會自動建立）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="資料集 split（預設: train）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大樣本數，None 表示全部（建議先用小量試跑）",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="17_0",
        help='Common Voice 版本，格式為 "17_0"（預設: 17_0）',
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="最短音訊時長（秒），過濾掉太短的樣本（預設: 1.0）",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="最長音訊時長（秒），過濾掉太長的樣本（預設: 30.0）",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安裝 datasets：pip install datasets")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 下載資料集
    dataset_name = f"mozilla-foundation/common_voice_{args.dataset_version}"
    print(f"正在從 Hugging Face 載入 {dataset_name} (zh-TW, split={args.split})...")
    print("（首次下載可能需要幾分鐘，資料會快取到 ~/.cache/huggingface/）")
    print("（若出現授權錯誤，請先執行：huggingface-cli login）")

    try:
        dataset = load_dataset(
            dataset_name,
            "zh-TW",
            split=args.split,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"\n錯誤：無法載入資料集：{e}")
        print("\n常見原因：")
        print("  1. 需要登入 Hugging Face：huggingface-cli login")
        print("  2. 需要在 Hugging Face 網頁同意 Common Voice 授權條款")
        print("  3. 網路問題（實驗室環境請確認可以連線 huggingface.co）")
        sys.exit(1)

    total_available = len(dataset)
    print(f"資料集共 {total_available} 筆可用")

    # 限制樣本數（隨機取樣確保多樣性）
    if args.max_samples and args.max_samples < total_available:
        dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
        print(f"隨機取樣 {args.max_samples} 筆（seed=42 確保可重現）")

    total_to_process = len(dataset)
    print(f"\n開始轉換 {total_to_process} 筆資料...")
    print(f"時長過濾：{args.min_duration}s - {args.max_duration}s")
    print(f"輸出目錄：{output_dir}\n")

    converted = 0
    skipped_empty = 0
    skipped_duration = 0

    for sample in tqdm(dataset, desc="轉換中", unit="筆"):
        # 快速過濾空文字（避免不必要的音訊處理）
        sentence = sample.get("sentence", "").strip()
        if not sentence:
            skipped_empty += 1
            continue

        # 嘗試轉換
        result = convert_sample(sample, output_dir, converted)
        if result is not None:
            converted += 1
        else:
            skipped_duration += 1

    # 統計摘要
    print(f"\n{'='*50}")
    print(f"轉換完成！")
    print(f"  成功轉換: {converted} 筆")
    print(f"  跳過（空文字）: {skipped_empty} 筆")
    print(f"  跳過（時長不符）: {skipped_duration} 筆")
    print(f"  輸出目錄: {output_dir}")
    print(f"{'='*50}")

    if converted == 0:
        print("\n警告：沒有成功轉換任何樣本，請檢查資料集是否正確載入。")
        sys.exit(1)

    # 儲存轉換統計（供後續參考）
    stats = {
        "dataset": dataset_name,
        "language": "zh-TW",
        "split": args.split,
        "total_available": total_available,
        "processed": total_to_process,
        "converted": converted,
        "skipped_empty": skipped_empty,
        "skipped_duration": skipped_duration,
        "filter_min_duration": args.min_duration,
        "filter_max_duration": args.max_duration,
    }
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n統計資訊已儲存至: {stats_path}")
    print(f"\n下一步：執行訓練腳本")
    print(f"  bash scripts/train_zh_tw.sh --data_dir={output_dir}")


if __name__ == "__main__":
    main()
