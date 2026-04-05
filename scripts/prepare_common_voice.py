"""
Mozilla Data Collective (MDC) Common Voice zh-TW 下載與格式轉換腳本。

取代原本的 HuggingFace 版本（已下架），改從 Mozilla 官方 API 下載。

用法：
    # 完整下載
    python3 scripts/prepare_common_voice.py \\
        --api_key YOUR_API_KEY \\
        --output_dir data/zh-tw-train

    # 下載後限制樣本數（隨機取樣）
    python3 scripts/prepare_common_voice.py \\
        --api_key YOUR_API_KEY \\
        --output_dir data/zh-tw-train \\
        --split train \\
        --max_samples 20000

    # 準備測試集
    python3 scripts/prepare_common_voice.py \\
        --api_key YOUR_API_KEY \\
        --output_dir data/zh-tw-test \\
        --split test \\
        --max_samples 500

    # 若已下載 tar.gz，跳過下載直接解析
    python3 scripts/prepare_common_voice.py \\
        --tarball "Common Voice Scripted Speech 25.0 - Chinese (Taiwan).tar.gz" \\
        --output_dir data/zh-tw-train \\
        --split train

MDC API 資訊：
    - 資料集 ID：cmn2g7eaj01fio10769r1m96n
    - API 文件：https://datacollective.mozillafoundation.org
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


# MDC API 設定
MDC_DATASET_ID = "cmn2g7eaj01fio10769r1m96n"
MDC_API_BASE = "https://datacollective.mozillafoundation.org"
MDC_DOWNLOAD_ENDPOINT = f"{MDC_API_BASE}/api/datasets/{MDC_DATASET_ID}/download"

# Common Voice tar.gz 解壓後的目錄結構
# zh-TW/
#   clips/         ← MP3 音訊檔
#   train.tsv      ← 訓練集（欄位：client_id, path, sentence, ...）
#   test.tsv       ← 測試集
#   validated.tsv  ← 已驗證（train + test + dev）
#   dev.tsv        ← 驗證集


def get_download_url(api_key: str) -> str:
    """
    向 MDC API 取得下載 URL。

    Args:
        api_key: MDC API Key。

    Returns:
        下載 URL 字串。
    """
    import urllib.request
    import json

    req = urllib.request.Request(
        url=MDC_DOWNLOAD_ENDPOINT,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"API 回應錯誤 HTTP {e.code}：{body}\n"
            f"請確認 API Key 正確，且已在 Mozilla Data Collective 網站同意授權條款。"
        )

    download_url = data.get("downloadUrl") or data.get("download_url")
    if not download_url:
        raise RuntimeError(
            f"API 回應中找不到 downloadUrl，完整回應：{data}"
        )

    return download_url


def download_with_progress(url: str, output_path: str) -> None:
    """
    下載檔案並顯示進度。

    Args:
        url: 下載 URL。
        output_path: 儲存路徑。
    """
    print(f"下載目標：{output_path}")

    def _progress_hook(block_count, block_size, total_size):
        if total_size > 0:
            downloaded = block_count * block_size
            percent = min(downloaded / total_size * 100, 100)
            downloaded_gb = downloaded / 1024 ** 3
            total_gb = total_size / 1024 ** 3
            bar_len = 40
            filled = int(bar_len * downloaded / total_size)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r  [{bar}] {percent:.1f}% {downloaded_gb:.2f}/{total_gb:.2f} GB",
                end="",
                flush=True,
            )
        else:
            downloaded = block_count * block_size
            print(
                f"\r  已下載 {downloaded / 1024 ** 2:.1f} MB",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(url, output_path, reporthook=_progress_hook)
    print()  # 換行


def parse_tsv_from_tar(
    tarball_path: str,
    split: str,
) -> list[dict]:
    """
    從 tar.gz 中讀取指定 split 的 TSV 資料。

    Common Voice TSV 欄位（依版本可能略有不同）：
        client_id, path, sentence, up_votes, down_votes,
        age, gender, accents, variant, locale, segment

    Args:
        tarball_path: tar.gz 路徑。
        split: "train"、"test"、"dev" 或 "validated"。

    Returns:
        TSV row 的 list，每筆包含 path 和 sentence。
    """
    tsv_target = f"zh-TW/{split}.tsv"
    rows = []

    print(f"  讀取 {tsv_target}...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        try:
            tsv_member = tar.getmember(tsv_target)
        except KeyError:
            # 有些版本的路徑可能略有不同
            members = tar.getnames()
            candidates = [m for m in members if m.endswith(f"{split}.tsv")]
            if not candidates:
                raise FileNotFoundError(
                    f"找不到 {tsv_target}，tar.gz 中的 .tsv 檔案：{[m for m in members if m.endswith('.tsv')]}"
                )
            tsv_target = candidates[0]
            tsv_member = tar.getmember(tsv_target)

        with tar.extractfile(tsv_member) as f:
            reader = csv.DictReader(
                io.TextIOWrapper(f, encoding="utf-8"),
                delimiter="\t",
            )
            for row in reader:
                sentence = row.get("sentence", "").strip()
                path = row.get("path", "").strip()
                if sentence and path:
                    rows.append({"path": path, "sentence": sentence})

    print(f"  TSV 共 {len(rows)} 筆")
    return rows


def extract_and_convert(
    tarball_path: str,
    rows: list[dict],
    output_dir: Path,
    max_samples: int | None = None,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> tuple[int, int]:
    """
    從 tar.gz 提取音訊並轉換為 VibeVoice 格式。

    Args:
        tarball_path: tar.gz 路徑。
        rows: TSV 資料列表。
        output_dir: 輸出目錄。
        max_samples: 最大樣本數。
        min_duration: 最小音訊時長（秒）。
        max_duration: 最大音訊時長（秒）。

    Returns:
        (converted, skipped) 統計數字。
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("需要安裝 soundfile：pip install soundfile")

    # 隨機取樣
    if max_samples and max_samples < len(rows):
        random.seed(42)
        rows = random.sample(rows, max_samples)
        print(f"  隨機取樣 {max_samples} 筆（seed=42）")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    print(f"  解壓並轉換 {len(rows)} 筆資料...")

    with tarfile.open(tarball_path, "r:gz") as tar:
        for i, row in enumerate(rows):
            # 顯示進度
            if i % 500 == 0:
                print(f"  進度：{i}/{len(rows)} ({i/len(rows)*100:.1f}%)", end="\r")

            clip_path_in_tar = f"zh-TW/clips/{row['path']}"
            sentence = row["sentence"]

            try:
                member = tar.getmember(clip_path_in_tar)
            except KeyError:
                skipped += 1
                continue

            # 提取音訊到記憶體
            with tar.extractfile(member) as audio_f:
                audio_bytes = audio_f.read()

            # 用 soundfile 讀取音訊資訊（MP3 需要 soundfile + cffi backend）
            try:
                with io.BytesIO(audio_bytes) as buf:
                    data, samplerate = sf.read(buf)
                duration = len(data) / samplerate
            except Exception:
                skipped += 1
                continue

            # 時長過濾
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue

            # 儲存音訊為 WAV
            audio_filename = f"{converted:06d}.wav"
            audio_out_path = output_dir / audio_filename

            try:
                with io.BytesIO(audio_bytes) as buf:
                    data, samplerate = sf.read(buf)
                sf.write(str(audio_out_path), data, samplerate)
            except Exception:
                skipped += 1
                continue

            # 建立 VibeVoice 格式 JSON
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
                "customized_context": ["繁體中文", "Traditional Chinese", "zh-TW"],
            }

            json_path = output_dir / f"{converted:06d}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)

            converted += 1

    print(f"\n  轉換完成：{converted} 筆成功，{skipped} 筆跳過")
    return converted, skipped


def main():
    parser = argparse.ArgumentParser(
        description="從 Mozilla Data Collective 下載 Common Voice zh-TW 並轉換為 VibeVoice 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：

  # 完整流程（下載 + 轉換訓練集，取 20000 筆）
  python3 scripts/prepare_common_voice.py \\
      --api_key 你的API_KEY \\
      --output_dir data/zh-tw-train \\
      --split train \\
      --max_samples 20000

  # 只準備測試集
  python3 scripts/prepare_common_voice.py \\
      --api_key 你的API_KEY \\
      --output_dir data/zh-tw-test \\
      --split test \\
      --max_samples 500

  # 已有 tar.gz 時跳過下載
  python3 scripts/prepare_common_voice.py \\
      --tarball "Common Voice Scripted Speech 25.0 - Chinese (Taiwan).tar.gz" \\
      --output_dir data/zh-tw-train \\
      --split train \\
      --max_samples 20000

注意：
  - tar.gz 約 2-5 GB，下載需要 10-30 分鐘（視網速）
  - 轉換過程需要額外磁碟空間（約 tar.gz 的 1.5 倍）
  - API Key 請勿寫死在腳本中，建議用環境變數：
      export MDC_API_KEY=你的KEY
      python3 scripts/prepare_common_voice.py --api_key $MDC_API_KEY ...
""",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("MDC_API_KEY"),
        help="Mozilla Data Collective API Key（或設定環境變數 MDC_API_KEY）",
    )
    parser.add_argument(
        "--tarball",
        type=str,
        default=None,
        help="已下載的 tar.gz 路徑（指定此項則跳過下載）",
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
        choices=["train", "test", "dev", "validated"],
        help="資料集 split（預設: train）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大樣本數（None = 全部）",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="最短音訊時長秒數（預設: 1.0）",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="最長音訊時長秒數（預設: 30.0）",
    )
    parser.add_argument(
        "--keep_tarball",
        action="store_true",
        help="轉換完成後保留原始 tar.gz（預設會刪除以節省空間）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Common Voice zh-TW 下載與格式轉換")
    print("=" * 60)
    print(f"  Split:      {args.split}")
    print(f"  輸出目錄:   {output_dir}")
    print(f"  最大樣本:   {args.max_samples or '全部'}")
    print(f"  時長過濾:   {args.min_duration}s - {args.max_duration}s")

    # ─── 決定 tarball 路徑 ────────────────────────────────────────────────
    tarball_path = args.tarball
    temp_tarball = None

    if tarball_path and os.path.exists(tarball_path):
        print(f"\n[1/3] 使用已存在的 tar.gz：{tarball_path}")
    else:
        # 需要下載
        if not args.api_key:
            print("\n錯誤：需要提供 --api_key 或設定環境變數 MDC_API_KEY")
            print("  取得 API Key：https://datacollective.mozillafoundation.org/profile")
            sys.exit(1)

        tarball_path = "common_voice_zh_tw.tar.gz"

        print(f"\n[1/3] 取得下載 URL...")
        try:
            download_url = get_download_url(args.api_key)
            print(f"  ✅ 取得下載 URL")
        except RuntimeError as e:
            print(f"\n❌ 錯誤：{e}")
            sys.exit(1)

        print(f"\n[2/3] 下載 tar.gz...")
        print(f"  （大型檔案，請耐心等待。可用 Ctrl+C 中斷後以 --tarball 重試）")
        try:
            download_with_progress(download_url, tarball_path)
            print(f"  ✅ 下載完成：{tarball_path}")
        except KeyboardInterrupt:
            print(f"\n\n中斷！已部分下載的檔案：{tarball_path}")
            print(f"若要繼續，請先刪除部分下載的檔案再重試。")
            sys.exit(1)

    # ─── 解析 TSV ────────────────────────────────────────────────────────
    step = "[3/3]" if not args.tarball else "[1/2]"
    print(f"\n{step} 讀取 {args.split}.tsv 並轉換...")

    try:
        rows = parse_tsv_from_tar(tarball_path, args.split)
    except FileNotFoundError as e:
        print(f"\n❌ 錯誤：{e}")
        sys.exit(1)

    if not rows:
        print("\n❌ 錯誤：TSV 中沒有可用的資料")
        sys.exit(1)

    # ─── 轉換 ────────────────────────────────────────────────────────────
    converted, skipped = extract_and_convert(
        tarball_path=tarball_path,
        rows=rows,
        output_dir=output_dir,
        max_samples=args.max_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    if converted == 0:
        print("\n❌ 沒有成功轉換任何樣本，請檢查 tar.gz 格式")
        sys.exit(1)

    # ─── 儲存統計 ────────────────────────────────────────────────────────
    stats = {
        "source": "Mozilla Data Collective",
        "dataset_id": MDC_DATASET_ID,
        "split": args.split,
        "language": "zh-TW",
        "total_in_tsv": len(rows),
        "converted": converted,
        "skipped": skipped,
        "filter_min_duration": args.min_duration,
        "filter_max_duration": args.max_duration,
    }
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ─── 刪除 tarball（可選）────────────────────────────────────────────
    if not args.keep_tarball and not args.tarball:
        print(f"\n清理 tar.gz（節省磁碟空間）...")
        os.remove(tarball_path)
        print(f"  已刪除：{tarball_path}")
        print(f"  （若需重新處理，請用 --api_key 重新下載）")

    # ─── 完成 ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"  成功: {converted} 筆")
    print(f"  跳過: {skipped} 筆")
    print(f"  輸出: {output_dir}")
    print(f"{'='*60}")
    print(f"\n下一步：執行訓練")
    print(f"  bash scripts/train_zh_tw.sh --data_dir={output_dir}")


if __name__ == "__main__":
    main()
