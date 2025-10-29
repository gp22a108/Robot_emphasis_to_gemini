import cv2
import time
import os

def take_picture(frame, delay_seconds=0):
    """
    指定された秒数待機した後に、与えられたフレームを画像として保存します。

    Args:
        frame: 保存する画像フレーム。
        delay_seconds (int): 撮影までの待機時間（秒）。
    """
    if frame is None:
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

    # 絶対パスを指定して画像を保存
    cv2.imwrite(filepath, frame)
    print(f"写真を{filepath}として保存しました。")
