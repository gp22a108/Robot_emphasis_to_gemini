import csv
import datetime
import os
import traceback

LOG_DIR = "logs"


def _ensure_dir():
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except Exception as e:
            print(f"[Logger] ログ用ディレクトリの作成に失敗しました: {e}")


def _append_csv(file_path, header, row, log_name):
    try:
        _ensure_dir()
        write_header = not os.path.exists(file_path)
        with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except PermissionError:
        print(f"[Logger Error] 権限がないため '{file_path}' に書き込めません。Excelなどで開いていないか確認してください。")
    except Exception as e:
        print(f"[Logger Error] {log_name} への書き込みに失敗しました: {e}")


def log_yolo_event(message, person_count=1):
    """YOLOの検出イベントをCSVに記録する。"""
    file_path = os.path.join(LOG_DIR, "yolo_log.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_csv(
        file_path,
        ["日時", "イベント", "人数"],
        [timestamp, message, person_count],
        "yolo_log.csv",
    )


def log_gemini_conversation(speaker, message):
    """Geminiとの会話ログをCSVに記録する。"""
    file_path = os.path.join(LOG_DIR, "gemini_log.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_csv(
        file_path,
        ["日時", "話者", "内容"],
        [timestamp, speaker, message],
        "gemini_log.csv",
    )


def log_interaction_result(result):
    """インタラクション結果をCSVに記録する。"""
    file_path = os.path.join(LOG_DIR, "interaction_log.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_csv(
        file_path,
        ["日時", "結果"],
        [timestamp, result],
        "interaction_log.csv",
    )


def log_system_error(context, exc=None, message=None):
    """システムエラーをCSVに記録する。"""
    log_system_event("ERROR", context, exc=exc, message=message)


def log_system_event(level, context, message=None, exc=None):
    """システムイベントをCSVに記録する。"""
    file_path = os.path.join(LOG_DIR, "system_error_log.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_type = type(exc).__name__ if exc else ""
    error_message = message if message is not None else (str(exc) if exc else "")
    trace = ""
    if exc:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    _append_csv(
        file_path,
        ["日時", "レベル", "コンテキスト", "例外種別", "メッセージ", "トレースバック"],
        [timestamp, level, context or "不明", error_type, error_message, trace],
        "system_error_log.csv",
    )
