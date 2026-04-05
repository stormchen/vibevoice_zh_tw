"""
VibeVoice-ASR 繁體中文推論腳本。

整合 LoRA 微調模型 + OpenCC 後處理的完整推論管道。

用法：
    # 用 LoRA 推論（帶後處理）
    python scripts/inference_zh_tw.py \\
        --lora_path ./output/zh-tw-lora \\
        --audio_file test.wav

    # 用已合併的獨立模型推論
    python scripts/inference_zh_tw.py \\
        --model_path ./models/vibevoice-asr-zh-tw \\
        --audio_file test.wav

    # 停用後處理（直接看模型原始輸出）
    python scripts/inference_zh_tw.py \\
        --lora_path ./output/zh-tw-lora \\
        --audio_file test.wav \\
        --no_postprocess

    # 加入熱詞提升辨識準確率
    python scripts/inference_zh_tw.py \\
        --lora_path ./output/zh-tw-lora \\
        --audio_file test.wav \\
        --context_info "人工智慧, 語音辨識, 台灣"

前置需求：
    pip install -e .                          # VibeVoice
    pip install peft                          # LoRA 推論
    pip install opencc-python-reimplemented   # 後處理（可選）
"""

import argparse
import json
import os
import sys
import time

import torch

# 確保能正確 import vibevoice 和 scripts 內的模組
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_ROOT_DIR, _SCRIPTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_model(
    base_model_path: str,
    lora_path: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    載入模型（支援 LoRA 和已合併的獨立模型）。

    Args:
        base_model_path: 基礎模型路徑（HuggingFace ID 或本地路徑）。
        lora_path: LoRA 權重目錄路徑。若為 None 則直接使用 base_model_path。
        device: 推論裝置（"cuda"、"cpu"、"auto"）。
        dtype: 模型精度。

    Returns:
        Tuple[model, processor]
    """
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    print(f"載入 Processor：{base_model_path}")
    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )

    print(f"載入模型：{base_model_path}")
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
        print(f"載入 LoRA 權重：{lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    print("模型載入完成 ✅")
    return model, processor


def transcribe(
    model,
    processor,
    audio_path: str,
    max_new_tokens: int = 8192,
    context_info: str | None = None,
    device: str = "cuda",
    use_postprocess: bool = True,
) -> dict:
    """
    轉錄音訊並套用繁中後處理。

    Args:
        model: 已載入的 ASR 模型。
        processor: VibeVoiceASRProcessor。
        audio_path: 音訊檔案路徑（支援 WAV、MP3、FLAC 等）。
        max_new_tokens: 最大生成 token 數。
        context_info: 上下文資訊（熱詞、領域術語等），可提升辨識準確率。
        device: 推論裝置。
        use_postprocess: 是否套用 OpenCC 後處理。

    Returns:
        包含轉錄結果的 dict：
        - raw_text: 模型原始輸出
        - final_text: 後處理後的輸出
        - segments: 結構化片段列表
        - generation_time: 生成耗時（秒）
        - postprocessed: 是否已套用後處理
    """
    print(f"\n轉錄：{audio_path}")

    # 音訊前處理
    inputs = processor(
        audio=audio_path,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )

    # 移到正確裝置
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # 生成設定
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": False,  # Greedy decoding，確保可重現性
    }

    # 生成
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_config)
    generation_time = time.time() - start_time

    # 解碼
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    # 後處理（OpenCC 繁體轉換）
    final_text = raw_text
    actually_postprocessed = False
    if use_postprocess:
        try:
            from postprocess_zh_tw import TraditionalChinesePostProcessor
            pp = TraditionalChinesePostProcessor(config="s2twp")
            final_text = pp.convert_transcription_json(raw_text)
            actually_postprocessed = True
        except ImportError:
            print("  ⚠️  opencc-python-reimplemented 未安裝，跳過後處理")
            print("      安裝命令：pip install opencc-python-reimplemented")

    # 解析結構化輸出
    segments = []
    try:
        segments = processor.post_process_transcription(final_text)
    except Exception:
        pass  # 解析失敗時保留空列表

    return {
        "raw_text": raw_text,
        "final_text": final_text,
        "segments": segments,
        "generation_time": round(generation_time, 2),
        "postprocessed": actually_postprocessed,
    }


def print_result(result: dict) -> None:
    """格式化印出轉錄結果。"""
    print(f"\n{'='*60}")
    print(f"轉錄結果")
    print(f"  生成耗時：{result['generation_time']}s")
    print(f"  後處理：{'已套用 OpenCC s2twp' if result['postprocessed'] else '未套用'}")
    print(f"{'='*60}")

    segments = result.get("segments", [])
    if segments:
        print(f"\n共 {len(segments)} 個片段：\n")
        for seg in segments:
            start = seg.get("start_time", "?")
            end = seg.get("end_time", "?")
            speaker = seg.get("speaker_id", "?")
            text = seg.get("text", "")
            print(f"  [{start} → {end}] 說話者 {speaker}：")
            print(f"    {text}")
    else:
        print("\n轉錄文字：")
        print(f"  {result['final_text']}")

    if result["raw_text"] != result["final_text"]:
        print(f"\n--- 模型原始輸出 ---")
        raw_preview = result["raw_text"]
        if len(raw_preview) > 500:
            raw_preview = raw_preview[:500] + "..."
        print(raw_preview)


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice-ASR 繁體中文推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 基本用法
  python scripts/inference_zh_tw.py \\
      --lora_path ./output/zh-tw-lora \\
      --audio_file test.wav

  # 加入熱詞（建議用於專業術語或人名）
  python scripts/inference_zh_tw.py \\
      --lora_path ./output/zh-tw-lora \\
      --audio_file test.wav \\
      --context_info "台積電, 鴻海, 人工智慧"

  # 不用 LoRA，直接用合併後的獨立模型
  python scripts/inference_zh_tw.py \\
      --model_path ./models/vibevoice-asr-zh-tw \\
      --audio_file test.wav
""",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-ASR",
        help="基礎模型路徑（HuggingFace ID 或本地目錄）",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA 權重目錄路徑（若省略則使用純基礎模型）",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="音訊檔案路徑（支援 WAV、MP3、FLAC、M4A 等）",
    )
    parser.add_argument(
        "--context_info",
        type=str,
        default=None,
        help="上下文資訊，如熱詞或領域術語（用逗號分隔）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="最大生成 token 數（預設: 8192）",
    )
    parser.add_argument(
        "--no_postprocess",
        action="store_true",
        help="停用 OpenCC 後處理（直接輸出模型原始結果）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="將結果儲存為 JSON 檔案（可選）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推論裝置（預設: cuda 若可用，否則 cpu）",
    )
    args = parser.parse_args()

    # 檢查音訊檔案存在
    if not os.path.exists(args.audio_file):
        print(f"錯誤：找不到音訊檔案：{args.audio_file}")
        sys.exit(1)

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32

    # 載入模型
    model, processor = load_model(
        base_model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
    )

    # 執行轉錄
    result = transcribe(
        model=model,
        processor=processor,
        audio_path=args.audio_file,
        max_new_tokens=args.max_new_tokens,
        context_info=args.context_info,
        device=args.device,
        use_postprocess=not args.no_postprocess,
    )

    # 印出結果
    print_result(result)

    # 儲存 JSON（可選）
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n結果已儲存至：{args.output_json}")


if __name__ == "__main__":
    main()
