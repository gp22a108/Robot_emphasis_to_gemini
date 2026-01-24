import time
import os
import threading
import Logger

# 音声再生用のグローバル変数
_shutter_sound = None
_sound_initialized = False

def _init_sound():
    """音声再生の初期化を行う。pygameがあれば優先して使用する。"""
    global _shutter_sound, _sound_initialized
    if _sound_initialized:
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(script_dir, "audio", "Camera-Compact01-01(Shutter).mp3")
    
    if os.path.exists(audio_path):
        try:
            # pygameを試行 (低遅延・音量調整可能)
            import pygame
            # ミキサーの初期化（既に初期化されていても安全）
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            _shutter_sound = pygame.mixer.Sound(audio_path)
            _shutter_sound.set_volume(1.0) # 音量最大 (0.0 - 1.0)
        except ImportError:
            # pygameがインストールされていない場合は何もしない（take_picture内でフォールバック）
            pass
        except Exception as e:
            Logger.log_system_error("シャッター音初期化", e)
            print(f"Warning: Failed to init pygame sound: {e}")
    
    _sound_initialized = True

# モジュール読み込み時に初期化を試みる（遅延を防ぐため）
_init_sound()

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
    if _shutter_sound:
        # pygameでロード済みなら即座に再生（低遅延）
        _shutter_sound.play()
    else:
        # pygameがない場合は playsound を使用（フォールバック）
        audio_path = os.path.join(script_dir, "audio", "Camera-Compact01-01(Shutter).mp3")
        if os.path.exists(audio_path):
            def play_fallback():
                try:
                    try:
                        from playsound3 import playsound
                    except ImportError:
                        from playsound import playsound
                    playsound(audio_path)
                except ImportError:
                    print("Tip: Install 'pygame' for faster audio: pip install pygame")
                except Exception as e:
                    Logger.log_system_error("シャッター音再生", e)
                    print(f"Warning: Failed to play sound: {e}")
            
            threading.Thread(target=play_fallback, daemon=True).start()

    # 絶対パスを指定して画像を保存
    if not cv2.imwrite(filepath, frame):
        Logger.log_system_error("撮影保存", message=f"保存に失敗しました: {filepath}")
        print(f"写真の保存に失敗しました: {filepath}")
        return
    print(f"写真を{filepath}として保存しました。")
