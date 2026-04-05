"""
VibeVoice-ASR 繁體中文微調評估腳本。

比較以下三種設定在 Common Voice zh-TW test set 上的效果：
  1. Baseline：原始 VibeVoice-ASR 模型
  2. 微調模型：LoRA 微調後的模型
  3. 微調 + 後處理：LoRA 微調 + OpenCC s2twp

評估指標：
  - CER（Character Error Rate）：字元錯誤率，越低越好
  - 繁體比例：輸出為繁體字的比例，越高越好
  - 完全繁體率：每筆輸出都是繁體的比例

用法：
    # 評估 LoRA 微調模型（與 baseline 比較）
    python scripts/evaluate_zh_tw.py \\
        --lora_path ./output/zh-tw-lora \\
        --test_dir data/zh-tw-test \\
        --max_samples 100

    # 只評估 baseline（不用 LoRA）
    python scripts/evaluate_zh_tw.py \\
        --test_dir data/zh-tw-test \\
        --max_samples 50

    # 儲存詳細報告
    python scripts/evaluate_zh_tw.py \\
        --lora_path ./output/zh-tw-lora \\
        --test_dir data/zh-tw-test \\
        --output_report reports/eval_results.json

前置需求：
    pip install jiwer opencc-python-reimplemented
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_ROOT_DIR, _SCRIPTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from jiwer import cer as compute_cer
except ImportError:
    print("錯誤：需要安裝 jiwer：pip install jiwer")
    sys.exit(1)


def load_test_data(test_dir: str, max_samples: int | None = None) -> list[dict]:
    """
    從目錄載入測試資料（VibeVoice 格式）。

    Args:
        test_dir: 測試資料目錄（由 prepare_common_voice.py 產生）。
        max_samples: 最大樣本數限制。

    Returns:
        樣本列表，每筆包含 audio_path 和 reference 文字。
    """
    test_path = Path(test_dir)
    samples = []

    json_files = sorted(test_path.glob("*.json"))
    for json_path in json_files:
        # 跳過統計檔案
        if json_path.name == "dataset_stats.json":
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️  跳過損壞的檔案 {json_path.name}：{e}")
            continue

        audio_path = test_path / data.get("audio_path", "")
        if not audio_path.exists():
            continue

        # 拼接所有片段文字作為參考答案
        reference_text = " ".join(
            seg.get("text", "").strip()
            for seg in data.get("segments", [])
            if seg.get("text", "").strip()
        )

        if not reference_text:
            continue

        samples.append({
            "audio_path": str(audio_path),
            "reference": reference_text,
            "duration": data.get("audio_duration", 0.0),
        })

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    return samples


def run_inference_batch(
    model,
    processor,
    samples: list[dict],
    device: str = "cuda",
    use_postprocess: bool = True,
    desc: str = "評估中",
) -> list[dict]:
    """
    對一批樣本執行推論並收集結果。

    Args:
        model: ASR 模型。
        processor: VibeVoiceASRProcessor。
        samples: 測試樣本列表。
        device: 推論裝置。
        use_postprocess: 是否套用 OpenCC 後處理。
        desc: tqdm 進度條描述。

    Returns:
        結果列表，每筆包含 reference、hypothesis 和 simplified_ratio。
    """
    from inference_zh_tw import transcribe
    from postprocess_zh_tw import TraditionalChinesePostProcessor

    results = []
    errors = 0

    for sample in tqdm(samples, desc=desc, unit="筆"):
        try:
            result = transcribe(
                model=model,
                processor=processor,
                audio_path=sample["audio_path"],
                device=device,
                use_postprocess=use_postprocess,
            )

            # 從 segments 提取假設文字
            if result["segments"]:
                hyp_text = " ".join(
                    seg.get("text", "").strip()
                    for seg in result["segments"]
                    if seg.get("text", "").strip()
                )
            else:
                hyp_text = result["final_text"].strip()

            simplified_ratio = TraditionalChinesePostProcessor.detect_simplified_ratio(
                hyp_text
            )

            results.append({
                "audio_path": sample["audio_path"],
                "reference": sample["reference"],
                "hypothesis": hyp_text,
                "simplified_ratio": simplified_ratio,
                "generation_time": result["generation_time"],
                "postprocessed": result["postprocessed"],
            })

        except Exception as e:
            errors += 1
            if errors <= 3:  # 只顯示前 3 個錯誤，避免刷屏
                print(f"  ⚠️  推論失敗 {os.path.basename(sample['audio_path'])}：{e}")

    if errors > 0:
        print(f"  ⚠️  共 {errors} 筆推論失敗")

    return results


def compute_metrics(results: list[dict]) -> dict:
    """
    由推論結果計算評估指標。

    Args:
        results: run_inference_batch 的輸出。

    Returns:
        指標 dict。
    """
    if not results:
        return {"error": "沒有可計算的結果"}

    references = [r["reference"] for r in results]
    hypotheses = [r["hypothesis"] for r in results]
    simplified_ratios = [r["simplified_ratio"] for r in results]
    times = [r["generation_time"] for r in results]

    overall_cer = compute_cer(references, hypotheses)
    avg_simplified = sum(simplified_ratios) / len(simplified_ratios)

    # 完全繁體率：簡體比例 < 1% 的樣本比例
    fully_traditional = sum(1 for r in simplified_ratios if r < 0.01)
    traditional_rate = fully_traditional / len(simplified_ratios)

    return {
        "num_samples": len(results),
        "cer_percent": round(overall_cer * 100, 2),
        "avg_simplified_percent": round(avg_simplified * 100, 2),
        "traditional_rate_percent": round(traditional_rate * 100, 2),
        "fully_traditional_count": fully_traditional,
        "total_time_sec": round(sum(times), 2),
        "avg_time_per_sample_sec": round(sum(times) / len(times), 2),
    }


def print_metrics(metrics: dict, label: str) -> None:
    """格式化印出指標。"""
    print(f"\n  📊 {label}")
    print(f"  {'─'*40}")
    if "error" in metrics:
        print(f"    ❌ {metrics['error']}")
        return
    print(f"    樣本數:           {metrics['num_samples']}")
    print(f"    CER:              {metrics['cer_percent']}%  {'↓ 越低越好' if True else ''}")
    print(f"    簡體殘留率:       {metrics['avg_simplified_percent']}%  {'↓ 越低越好' if True else ''}")
    print(f"    完全繁體率:       {metrics['traditional_rate_percent']}%  {'↑ 越高越好' if True else ''}")
    print(f"    每筆平均耗時:     {metrics['avg_time_per_sample_sec']}s")


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice-ASR 繁體中文微調效果評估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
建議流程：
  1. 先準備測試資料：
     python scripts/prepare_common_voice.py --output_dir data/zh-tw-test --split test --max_samples 200

  2. 執行評估（先用 --max_samples 50 快速確認腳本可運行）：
     python scripts/evaluate_zh_tw.py \\
         --lora_path ./output/zh-tw-lora \\
         --test_dir data/zh-tw-test \\
         --max_samples 50

  3. 完整評估並儲存報告：
     python scripts/evaluate_zh_tw.py \\
         --lora_path ./output/zh-tw-lora \\
         --test_dir data/zh-tw-test \\
         --output_report reports/eval_$(date +%Y%m%d).json
""",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="基礎模型路徑（預設: microsoft/VibeVoice-ASR）",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA 權重目錄（不指定則只評估 baseline）",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="測試資料目錄（由 prepare_common_voice.py 產生）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大測試樣本數（建議先用 50-100 快速驗證）",
    )
    parser.add_argument(
        "--no_postprocess",
        action="store_true",
        help="停用 OpenCC 後處理（只評估模型原始輸出）",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="跳過 baseline 評估（節省時間，直接評估微調模型）",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="評估報告 JSON 輸出路徑（可選）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # ─── 資料載入 ──────────────────────────────────────────────────────────
    print(f"載入測試資料：{args.test_dir}")
    samples = load_test_data(args.test_dir, args.max_samples)
    if not samples:
        print("錯誤：沒有可用的測試資料")
        print("請先執行：python scripts/prepare_common_voice.py --output_dir <test_dir> --split test")
        sys.exit(1)
    print(f"共 {len(samples)} 筆測試樣本")

    from inference_zh_tw import load_model

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    all_metrics = {}

    # ─── Baseline 評估 ─────────────────────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print("第 1/2 步：評估 Baseline（原始 VibeVoice-ASR）")
        print(f"{'='*60}")

        baseline_model, baseline_processor = load_model(
            base_model_path=args.model_path,
            lora_path=None,
            device=args.device,
            dtype=dtype,
        )

        baseline_results = run_inference_batch(
            baseline_model,
            baseline_processor,
            samples,
            device=args.device,
            use_postprocess=False,  # Baseline 不用後處理，看原始輸出
            desc="Baseline 推論",
        )

        all_metrics["baseline"] = compute_metrics(baseline_results)
        del baseline_model  # 釋放記憶體

    # ─── 微調模型評估 ──────────────────────────────────────────────────────
    if args.lora_path:
        print(f"\n{'='*60}")
        step = "第 2/2 步" if not args.skip_baseline else "評估"
        print(f"{step}：評估微調模型（LoRA）")
        print(f"{'='*60}")

        finetuned_model, finetuned_processor = load_model(
            base_model_path=args.model_path,
            lora_path=args.lora_path,
            device=args.device,
            dtype=dtype,
        )

        # 不含後處理
        ft_results_raw = run_inference_batch(
            finetuned_model,
            finetuned_processor,
            samples,
            device=args.device,
            use_postprocess=False,
            desc="微調模型推論（無後處理）",
        )
        all_metrics["finetuned_raw"] = compute_metrics(ft_results_raw)

        # 含後處理
        if not args.no_postprocess:
            ft_results_pp = run_inference_batch(
                finetuned_model,
                finetuned_processor,
                samples,
                device=args.device,
                use_postprocess=True,
                desc="微調模型推論（含後處理）",
            )
            all_metrics["finetuned_postprocessed"] = compute_metrics(ft_results_pp)

    # ─── 輸出結果 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("評估結果總覽")
    print(f"{'='*60}")

    if "baseline" in all_metrics:
        print_metrics(all_metrics["baseline"], "Baseline（原始模型，無後處理）")

    if "finetuned_raw" in all_metrics:
        print_metrics(all_metrics["finetuned_raw"], "微調模型（無後處理）")

    if "finetuned_postprocessed" in all_metrics:
        print_metrics(all_metrics["finetuned_postprocessed"], "微調模型 + OpenCC 後處理")

    # 如果有 baseline 和微調，顯示改善幅度
    if "baseline" in all_metrics and "finetuned_raw" in all_metrics:
        bl = all_metrics["baseline"]
        ft = all_metrics["finetuned_raw"]
        if "cer_percent" in bl and "cer_percent" in ft:
            cer_delta = bl["cer_percent"] - ft["cer_percent"]
            trad_delta = ft["traditional_rate_percent"] - bl.get("traditional_rate_percent", 0)
            print(f"\n  📈 微調效果（Baseline → 微調）：")
            print(f"    CER 改善：{cer_delta:+.2f}%（正值為進步）")
            print(f"    繁體率提升：{trad_delta:+.2f}%")

    print(f"\n{'='*60}")

    # ─── 儲存報告 ──────────────────────────────────────────────────────────
    if args.output_report:
        report = {
            "evaluation_config": {
                "model_path": args.model_path,
                "lora_path": args.lora_path,
                "test_dir": args.test_dir,
                "num_samples": len(samples),
                "device": args.device,
            },
            "metrics": all_metrics,
        }
        os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"報告已儲存至：{args.output_report}")


if __name__ == "__main__":
    main()
