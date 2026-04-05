"""
繁體中文後處理模組。

提供 OpenCC 簡繁轉換和繁體品質檢測功能。
可獨立使用也可被其他腳本 import。

用法（獨立執行）：
    python scripts/postprocess_zh_tw.py --text "这个软件很好用"
    python scripts/postprocess_zh_tw.py --text "今天的内存使用率很高"
    python scripts/postprocess_zh_tw.py --detect-only --text "这个软件"
    echo "这是测试" | python scripts/postprocess_zh_tw.py
"""

import argparse
import json
import sys


def _check_opencc():
    """檢查 opencc 是否已安裝。"""
    try:
        import opencc  # noqa: F401
        return True
    except ImportError:
        return False


class TraditionalChinesePostProcessor:
    """
    繁體中文後處理器。

    使用 OpenCC 進行簡繁轉換，支援台灣用語詞彙替換。
    """

    # 常見簡體字集（用於快速偵測，涵蓋高頻率的簡繁差異字）
    # 注意：這只是近似偵測，不保證 100% 準確
    _SIMPLIFIED_CHARS = frozenset(
        "国际学语处见进还对问当应点长还来这们间从认经实间发现计没时发"
        "两规观项业权责种义样设过变动区现对问处关几发认来还这间进长见"
        "实计点应当对问间从认经现发时计没时样设过变动步样设过变动步"
        "软件内存处理器网络硬件显示器键盘鼠标程序计算机"
    )

    def __init__(self, config: str = "s2twp"):
        """
        初始化後處理器。

        Args:
            config: OpenCC 轉換設定：
                - s2twp：簡體 → 繁體（台灣用語 + 詞彙轉換）**推薦**
                - s2tw： 簡體 → 繁體（僅字形，不轉詞彙）
                - s2t：  簡體 → 繁體（基本轉換）

        Raises:
            ImportError: 若 opencc-python-reimplemented 未安裝。
        """
        if not _check_opencc():
            raise ImportError(
                "需要安裝 opencc-python-reimplemented：\n"
                "  pip install opencc-python-reimplemented"
            )
        import opencc
        self.converter = opencc.OpenCC(config)
        self.config = config

    def convert(self, text: str) -> str:
        """
        將文字轉換為繁體中文（台灣用語）。

        Args:
            text: 輸入文字（可為簡體或繁體）。

        Returns:
            轉換後的繁體中文文字。
        """
        if not text:
            return text
        return self.converter.convert(text)

    def convert_transcription_json(self, json_text: str) -> str:
        """
        轉換 VibeVoice ASR JSON 格式的轉錄結果。

        只轉換 "Content" 欄位的文字，完整保留 JSON 結構
        （包含時間戳、說話者 ID 等）。

        Args:
            json_text: VibeVoice ASR 輸出的 JSON 字串，或純文字。

        Returns:
            轉換後的 JSON 字串（或純文字）。
        """
        try:
            data = json.loads(json_text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "Content" in item:
                        item["Content"] = self.convert(item["Content"])
            elif isinstance(data, dict) and "Content" in data:
                data["Content"] = self.convert(data["Content"])
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            # 非 JSON 格式，直接轉換整段文字
            return self.convert(json_text)

    @staticmethod
    def detect_simplified_ratio(text: str) -> float:
        """
        偵測文字中簡體字的近似比例。

        使用常見簡體字集進行比對，結果為近似值。

        Args:
            text: 要分析的文字。

        Returns:
            0.0～1.0 的比例值（0.0 表示全繁體，1.0 表示全簡體）。
        """
        chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        if not chinese_chars:
            return 0.0

        # 計算出現在簡體字集中的字數
        simplified_count = sum(
            1 for c in chinese_chars
            if c in TraditionalChinesePostProcessor._SIMPLIFIED_CHARS
        )
        return simplified_count / len(chinese_chars)

    @staticmethod
    def is_traditional(text: str, threshold: float = 0.02) -> bool:
        """
        判斷文字是否為繁體中文。

        Args:
            text: 要判斷的文字。
            threshold: 簡體字比例門檻（低於此值視為繁體）。

        Returns:
            True 表示判斷為繁體，False 表示可能含有簡體字。
        """
        return TraditionalChinesePostProcessor.detect_simplified_ratio(text) < threshold


def main():
    parser = argparse.ArgumentParser(
        description="繁體中文後處理工具（OpenCC 簡繁轉換）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python scripts/postprocess_zh_tw.py --text "这个软件的内存使用率很高"
  # 輸出：這個軟體的記憶體使用率很高

  echo "这是测试" | python scripts/postprocess_zh_tw.py
  # 輸出：這是測試

  python scripts/postprocess_zh_tw.py --detect-only --text "这个软件"
  # 輸出：簡體字比例: 66.67%
""",
    )
    parser.add_argument("--text", type=str, help="要轉換的文字（或用 stdin）")
    parser.add_argument("--file", type=str, help="要轉換的檔案路徑")
    parser.add_argument(
        "--config",
        type=str,
        default="s2twp",
        choices=["s2twp", "s2tw", "s2t"],
        help="OpenCC 轉換設定（預設: s2twp，推薦）",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="只偵測簡體字比例，不進行轉換",
    )
    args = parser.parse_args()

    # 讀取輸入文字
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        parser.print_help()
        print("\n錯誤：請提供 --text、--file 或透過 stdin 輸入文字")
        sys.exit(1)

    if args.detect_only:
        ratio = TraditionalChinesePostProcessor.detect_simplified_ratio(text)
        is_trad = TraditionalChinesePostProcessor.is_traditional(text)
        print(f"簡體字比例: {ratio:.2%}")
        print(f"判定結果: {'繁體中文' if is_trad else '含有簡體字'}")
        return

    # 執行轉換
    processor = TraditionalChinesePostProcessor(config=args.config)
    result = processor.convert(text)
    print(result)


if __name__ == "__main__":
    main()
