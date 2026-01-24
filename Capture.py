import time
import os
import threading
import Logger

def _play_shutter_sound(audio_path: str):
    """シャッター音を再生する（playsound系を使用）。"""
    try:
        try:
            from playsound3 import playsound
        except ImportError:
            from playsound import playsound
        playsound(audio_path)
    except ImportError:
        print("Tip: Install 'playsound3' (or 'playsound') for shutter audio.")
    except Exception as e:
        Logger.log_system_error("シャッター音再生", e)
        print(f"Warning: Failed to play sound: {e}")

def take_picture(frame, delay_seconds=0):
    """
    指定された秒数待機した後に、与えられたフレームを画像として保存します。

    Args:
        frame: 保存する画像フレーム。
        delay_seconds (int): 撮影までの待機時間（秒）。
    """
    # 遅延インポート
    import cv2

    if frame is None:
        Logger.log_system_error("撮影", message="フレームが提供されなかったため保存できませんでした")
        print("フレームが提供されなかったため、写真を撮影できませんでした。")
        return

    if delay_seconds > 0:
        print(f"{delay_seconds}秒後に写真を撮影します。")
        time.sleep(delay_seconds)
    else:
        print("写真を撮影します。")

    # このスクリプトのディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 保存先フォルダを 'captures' に設定
    captures_dir = os.path.join(script_dir, 'captures')

    # 'captures' フォルダがなければ作成
    os.makedirs(captures_dir, exist_ok=True)

    # ファイル名を現在時刻で生成
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(captures_dir, filename)

    # --- シャッター音の再生 ---
    audio_path = os.path.join(script_dir, "audio", "Camera-Compact01-01(Shutter).mp3")
    if os.path.exists(audio_path):
        threading.Thread(
            target=_play_shutter_sound,
            args=(audio_path,),
            daemon=True
        ).start()

    # 絶対パスを指定して画像を保存
    if not cv2.imwrite(filepath, frame):
        Logger.log_system_error("撮影保存", message=f"保存に失敗しました: {filepath}")
        print(f"写真の保存に失敗しました: {filepath}")
        return
    print(f"写真を{filepath}として保存しました。")
