# VoiceTyper 繁體中文語音輸入桌面應用 — 設計文件

## 概述

VoiceTyper 是一款基於 Electron 的跨平台（macOS / Windows）語音輸入桌面應用。
使用者按下快捷鍵即可錄音，語音送到遠端 ASR 伺服器辨識後，自動將文字貼入游標位置。

## 核心體驗

常駐 Menu Bar / System Tray → 按快捷鍵錄音 → 語音送到 API → 辨識結果自動貼到游標位置

## 架構

```
Electron App
├── Menu Bar / System Tray ← 全域快捷鍵監聽
├── 錄音模組 (Web Audio API)
│   ├── Push-to-talk 模式
│   └── Toggle 模式
├── API 客戶端（可切換後端）
│   ├── Google Chirp 3 (STT V2)
│   └── vLLM Server (自架 VibeVoice-ASR)
└── 剪貼簿 + 模擬鍵盤 → 貼到游標位置
```

## 功能規格

### 1. 系統匣常駐
- macOS: Menu Bar 圖示
- Windows: System Tray 圖示
- 三種狀態：待機（灰）、錄音中（紅）、辨識中（橘）

### 2. 全域快捷鍵
- 預設：Cmd+Shift+V (macOS) / Ctrl+Shift+V (Windows)
- 可自訂
- Push-to-talk / Toggle 兩種模式可切換

### 3. 語音辨識後端（可切換）
- Google Chirp 3 (STT V2)：開發測試用，需填 API Key
- vLLM Server：正式使用自己的繁中模型，需填 API URL

### 4. 自動貼上
- 辨識結果寫入剪貼簿
- 模擬 Cmd+V / Ctrl+V 貼到當前游標位置

### 5. 設定面板
- 後端選擇、API 設定、快捷鍵、錄音模式、麥克風選擇

## 技術選型

- Electron + HTML/CSS/JS
- electron-builder 打包 .dmg + .exe
