# 任務追蹤：VibeVoice-ASR 繁體中文微調

| # | 任務 | 狀態 | 備註 |
|---|------|------|------|
| 1 | 資料準備腳本 (prepare_common_voice.py) | ✅ done | 語法驗證通過，完整錯誤處理 |
| 2 | 繁中後處理模組 (postprocess_zh_tw.py) | ✅ done | 4 項功能測試全通過 |
| 3 | 訓練啟動腳本 (train_zh_tw.sh) | ✅ done | Shell 語法驗證通過 |
| 4 | 推論腳本 (inference_zh_tw.py) | ✅ done | 語法驗證通過，支援 LoRA + 後處理 + JSON 輸出 |
| 5 | 評估腳本 (evaluate_zh_tw.py) | ✅ done | 支援 baseline vs 微調對比，CER + 繁體率 |
| 6 | LoRA 合併腳本 (merge_lora.py) | ✅ done | 含前置檢查、磁碟驗證、元資料儲存 |
| 7 | 實驗室操作指南 | ✅ done | 5 階段完整指南，含排錯 QA |
