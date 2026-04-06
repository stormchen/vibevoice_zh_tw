# VibeVoice-ASR MacOS Local Demo
import gradio as gr
import torch
import gc
import json
from pathlib import Path
import os
import sys

# 把 scripts 目錄加入路徑
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from inference_zh_tw import load_model, transcribe
except ImportError:
    print("找不到 inference_zh_tw 模組，請確保此腳本置於 scripts/ 目錄下。")
    sys.exit(1)

model = None
processor = None
MODEL_PATH = "models/vibevoice-asr-zh-tw"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def check_model_loaded():
    return model is not None and processor is not None

def load_model_action():
    global model, processor
    if check_model_loaded():
        return "狀態未變更：模型已經在記憶體中了 ✅"
        
    print("載入模型進入記憶體...")
    try:
        model, processor = load_model(
            base_model_path=MODEL_PATH,
            lora_path=None,
            device=DEVICE,
            dtype=torch.float16
        )
        return f"模型已載入 ✅ (硬體: {DEVICE})"
    except Exception as e:
        model, processor = None, None
        return f"載入失敗: {e}"

def unload_model_action():
    global model, processor
    if not check_model_loaded():
        return "狀態未變更：模型原本就沒有載入"
        
    print("釋放模型記憶體...")
    model = None
    processor = None
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
    return "模型記憶體已成功釋放 🗑️"

def process_audio(audio_path, context_info, use_pp):
    if not check_model_loaded():
        return "⚠️ 請先點擊上方的「載入模型」按鈕", ""
        
    if not audio_path:
        return "⚠️ 尚未錄音或選擇檔案", ""

    print(f"處理音訊: {audio_path}")
    try:
        # ASR 轉錄
        result = transcribe(
            model=model,
            processor=processor,
            audio_path=audio_path,
            max_new_tokens=4096,
            context_info=context_info if context_info.strip() else None,
            device=DEVICE,
            use_postprocess=use_pp
        )
        
        time_msg = f"生成耗時: {result['generation_time']} 秒\n"
        time_msg += f"後處理: {'已套用 OpenCC s2twp' if result['postprocessed'] else '未套用'}"
        return result["final_text"], time_msg
    except Exception as e:
        return f"發生錯誤: {e}", ""

with gr.Blocks(title="VibeVoice MacOS 本機版") as demo:
    gr.Markdown("# VibeVoice 繁體中文語音辨識 (Mac M4 專用)")
    gr.Markdown("專為 16GB 統一記憶體優化的本地推論介面，測試完畢可隨時釋放記憶體。")
    
    with gr.Row():
        load_btn = gr.Button("1️⃣ 載入模型 (Load Model)", variant="primary")
        unload_btn = gr.Button("🗑️ 卸載模型 (釋放記憶體)")
        
    status_text = gr.Markdown("狀態：目前模型**未載入**")
    
    load_btn.click(fn=load_model_action, outputs=[status_text])
    unload_btn.click(fn=unload_model_action, outputs=[status_text])
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="語音輸入 (麥克風錄音或上傳)")
            context_input = gr.Textbox(label="上下文提示詞 / 熱詞 (可選)", placeholder="如：人工智慧, 台灣, 捷運...")
            pp_checkbox = gr.Checkbox(label="啟用 OpenCC 繁體中文後處理 (s2twp)", value=True)
            submit_btn = gr.Button("📥 2️⃣ 開始語音辨識", variant="primary")
            
        with gr.Column():
            text_output = gr.Textbox(label="辨識結果", lines=10)
            meta_output = gr.Textbox(label="效能數據", lines=2)

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, context_input, pp_checkbox],
        outputs=[text_output, meta_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
