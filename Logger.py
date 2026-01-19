import os
import datetime
import csv

LOG_DIR = "logs"

def _ensure_dir():
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except Exception as e:
            print(f"[Logger] Failed to create log directory: {e}")

def log_yolo_event(message, person_count=1):
    """YOLOの検出イベントをログに記録"""
    file_path = os.path.join(LOG_DIR, "yolo_log.csv")
    try:
        _ensure_dir()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # ヘッダー書き込み（ファイルが新規の場合）
        write_header = not os.path.exists(file_path)
        
        # Excelで文字化けしないように utf-8-sig を使用
        with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Timestamp", "Event", "PersonCount"])
            writer.writerow([timestamp, message, person_count])
    except PermissionError:
        print(f"[Logger Error] Permission denied: '{file_path}'. Excelなどでファイルを開いている場合は閉じてください。")
    except Exception as e:
        print(f"[Logger Error] Failed to write to yolo_log.csv: {e}")

def log_gemini_conversation(speaker, message):
    """Geminiとの会話内容をログに記録"""
    file_path = os.path.join(LOG_DIR, "gemini_log.csv")
    try:
        _ensure_dir()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        write_header = not os.path.exists(file_path)
        
        # Excelで文字化けしないように utf-8-sig を使用
        with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Timestamp", "Speaker", "Message"])
            writer.writerow([timestamp, speaker, message])
    except PermissionError:
        print(f"[Logger Error] Permission denied: '{file_path}'. Excelなどでファイルを開いている場合は閉じてください。")
    except Exception as e:
        print(f"[Logger Error] Failed to write to gemini_log.csv: {e}")

def log_interaction_result(result):
    """インタラクションの結果（写真撮影の有無など）をログに記録"""
    file_path = os.path.join(LOG_DIR, "interaction_log.csv")
    try:
        _ensure_dir()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        write_header = not os.path.exists(file_path)
        
        # Excelで文字化けしないように utf-8-sig を使用
        with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Timestamp", "Result"])
            writer.writerow([timestamp, result])
    except PermissionError:
        print(f"[Logger Error] Permission denied: '{file_path}'. Excelなどでファイルを開いている場合は閉じてください。")
    except Exception as e:
        print(f"[Logger Error] Failed to write to interaction_log.csv: {e}")
