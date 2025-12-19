# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

pip install google-genai opencv-python pyaudio pillow mss aiohttp psutil

環境変数 GOOGLE_API_KEY を設定して実行してください。

python Gemini.py
python Gemini.py --mode screen
"""

import time
_t_start = time.perf_counter()
print(f"[Gemini] プロセス起動。インポートを開始します...")

import asyncio
import io
import traceback
import argparse

import aiohttp
import cv2
print(f"[Gemini] OpenCV/aiohttp インポート完了: {time.perf_counter() - _t_start:.3f}s")

import pyaudio
import PIL.Image
import mss
print(f"[Gemini] Audio/Imageライブラリ インポート完了: {time.perf_counter() - _t_start:.3f}s")

import config  # 設定ファイル

from google import genai
from google.genai import types
print(f"[Gemini] Google GenAI インポート完了 (準備完了): {time.perf_counter() - _t_start:.3f}s")

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1beta"})

# ネイティブ音声 Live モデル用設定
CONFIG = {
    "response_modalities": ["AUDIO"],       # TEXT ではなく AUDIO
    "output_audio_transcription": {},       # 音声出力を文字起こししてもらう
}

pya = pyaudio.PyAudio()


def list_audio_devices():
    """デバッグ用: 利用可能なマイク一覧を表示"""
    print("Available audio input devices:")
    for i in range(pya.get_device_count()):
        dev = pya.get_device_info_by_index(i)
        if dev["maxInputChannels"] > 0:
            print(f"  Index {i}: {dev['name']}")
    print("-" * 20)
    default_dev = pya.get_default_input_device_info()
    print(
        f"Default audio input device: "
        f"Index {default_dev['index']}: {default_dev['name']}"
    )
    print("-" * 20)


class AudioLoop:
    def __init__(self, video_mode: str = DEFAULT_MODE):
        self.video_mode = video_mode
        self.system_instruction = """### 役割と振る舞い
        - 必ず、テキストを出力してください。
        - ユーザーに対しては、友達のように親しみやすく、少し馴れ馴れしい口調（タメ口など）で接してください。
        - 基本的にすべて日本語で回答してください。
        - まず初めにカメラを見てユーザーの服装や身につけているものを見て褒めてください。その後にストリートスナップスナップやっていることを説明してください。
        - 最後に写真を取るように誘導してください。
        - だいたい会話が３往復目くらいで写真撮影してください。
        - 質問の後は会話を無理やり続けないでください。
        - 少し一つの会話の長さ控えめで
        - 写真撮影の同意を取ってください。

        ### 【重要】カメラ撮影の制御コマンド
        ユーザーから写真撮影やカメラの起動を依頼された場合（例：「写真撮って」「撮影して」「カメラ起動」など）は、以下の手順を**必ず**守ってください。

        1. 撮影の合図となるような、明るい返答をする（例：「いいよ、撮るね！」「はい、チーズ！」「撮るよ～」など）。
        2. **応答の最後（文末）**に、必ず `[CAPTURE_IMAGE]` という文字列を含める。
           ※この文字列はシステムがカメラを起動するためのトリガーです。絶対に省略や変更をしないでください。

        ### 応答例
        AI：「オッケー！いい顔してるね〜。はい、とるよー [CAPTURE_IMAGE]」"""
        self.out_queue: asyncio.Queue | None = None
        self.session = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.yolo_detector = None
        self.audio_stream = None

        self.mic_is_active = asyncio.Event()
        self.mic_is_active.set()

    @staticmethod
    def _user_turn(text: str) -> types.Content:
        # google-genai 1.55.0 の send_client_content(turns=...) は Content が必要
        return types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )

    def _handle_yolo_detection(self):
        """YOLO からの検出イベントを処理"""
        print("YOLO detection event received, sending to Gemini.")
        if self.session and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.session.send_client_content(
                    turns=self._user_turn("Detected"),
                    turn_complete=True,
                ),
                self.loop,
            )

    async def send_text(self):
        """標準入力からテキストを Live API に送信"""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await self.session.send_client_content(
                turns=self._user_turn(text or "."),
                turn_complete=True,
            )

    def _get_screen_jpeg_bytes(self) -> bytes:
        """画面キャプチャを 1 フレーム取得して JPEG bytes にする"""
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            grabbed = sct.grab(monitor)

        img = PIL.Image.frombytes("RGB", grabbed.size, grabbed.rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    async def get_screen(self):
        """画面キャプチャを連続で out_queue に投入（1fps）"""
        while True:
            jpeg_bytes = await asyncio.to_thread(self._get_screen_jpeg_bytes)
            await asyncio.sleep(1.0)
            await self.out_queue.put(
                ("video", types.Blob(data=jpeg_bytes, mime_type="image/jpeg"))
            )

    async def send_realtime(self):
        """out_queue から Live API にリアルタイム送信"""
        while True:
            kind, payload = await self.out_queue.get()

            # send_realtime_input は 1 回で 1 引数だけ送る
            if kind == "audio":
                await self.session.send_realtime_input(audio=payload)
            elif kind == "video":
                # 静止画フレーム（image/jpeg）を video フレームとして送る
                await self.session.send_realtime_input(video=payload)

    async def listen_audio(self):
        """マイクから音声を取得して out_queue に投入"""
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}

        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if self.mic_is_active.is_set():
                await self.out_queue.put(
                    (
                        "audio",
                        types.Blob(data=data, mime_type="audio/pcm;rate=16000"),
                    )
                )

    async def receive_responses(self):
        """Live API からのレスポンスを受信し、テキストを組み立てて処理"""
        from Voicevox_player import VoicevoxStreamPlayer  # 遅延インポート

        while True:
            player = None
            capture_callback_triggered = asyncio.Event()

            def capture_action():
                if not capture_callback_triggered.is_set():
                    print("--- 写真撮影タスクを非同期で開始します ---")
                    from Capture import take_picture  # 遅延インポート

                    frame_to_capture = self.yolo_detector.get_current_frame()
                    coro = asyncio.to_thread(take_picture, frame_to_capture, 0)
                    asyncio.run_coroutine_threadsafe(coro, self.loop)
                    capture_callback_triggered.set()

            try:
                # 1 ターン分のレスポンスストリーム
                turn = self.session.receive()
                has_received_text = False
                full_response_text = []

                async for response in turn:
                    text = None

                    # 将来 TEXT モデルを使う場合の互換: response.text
                    if getattr(response, "text", None):
                        text = response.text

                    # native audio + transcription 用
                    server_content = getattr(response, "server_content", None)
                    if server_content is not None:
                        output_tx = getattr(server_content, "output_transcription", None)
                        if output_tx is not None and getattr(output_tx, "text", None):
                            text = output_tx.text

                    if not text:
                        continue

                    if not has_received_text:
                        has_received_text = True
                        # 最初のテキストを受け取ったらマイクを停止し、再生プレーヤーを初期化
                        self.mic_is_active.clear()
                        print("\n--- Mic paused while speaking ---")
                        if self.yolo_detector:
                            self.yolo_detector.pause()
                        
                        player = VoicevoxStreamPlayer(
                            speaker=config.SPEAKER_ID,
                            on_last_chunk_start=None # ストリーミング中はコールバックを直接制御
                        )
                        if not player.is_connected:
                            print(f"\n[Warning] Voicevoxへの接続に失敗しました。音声再生をスキップします。")
                            player = None

                    full_response_text.append(text)
                    
                    # CAPTURE_IMAGEトリガーを検出し、再生用のテキストから削除
                    if "[CAPTURE_IMAGE]" in text:
                        print("\n!!! 撮影トリガーを検出しました !!!")
                        text = text.replace("[CAPTURE_IMAGE]", "").strip()
                        
                        if player:
                            # 最後のチャンク再生開始時に撮影を実行するように設定
                            player.on_last_chunk_start = capture_action
                        else:
                            # 音声再生がない場合は即座に撮影
                            capture_action()

                    if player and text:
                        player.add_text(text)

                if has_received_text:
                    # 応答テキスト全体を構築して表示 (音声再生完了を待たずに表示)
                    final_text = "".join(full_response_text).replace("[CAPTURE_IMAGE]", "").strip()
                    print("\n\n--- [Full Response Captured] ---")
                    print(final_text)
                    print("--------------------------------\n")

                    if player:
                        player.finish()
                        await asyncio.to_thread(player.wait_done)
                        print("\n[Gemini] 音声再生が完了しました。")


            finally:
                # ターン終了後、マイクとYOLOを再開
                self.mic_is_active.set()
                print("--- Mic resumed ---")
                if self.yolo_detector:
                    self.yolo_detector.resume()
                if player:
                    player.close()


    async def run(self):
        """メインのタスクグループを立ち上げる"""
        self.loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        c0 = time.process_time()
        try:
            print("[Gemini] YOLOモジュールを遅延インポート中...")
            from YOLO import YOLOOptimizer
            print(f"[Gemini] YOLOモジュール インポート完了: {time.perf_counter() - _t_start:.3f}s")

            # YOLO 検出器を初期化して開始
            self.yolo_detector = YOLOOptimizer(on_detection=self._handle_yolo_detection)
            self.yolo_detector.start()

            live_config = CONFIG.copy()
            live_config["system_instruction"] = self.system_instruction

            async with (
                client.aio.live.connect(model=MODEL, config=live_config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "screen":
                    tg.create_task(self.get_screen())
                tg.create_task(self.receive_responses())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except Exception:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exc()
        finally:
            if self.yolo_detector:
                self.yolo_detector.stop()
            elapsed_wall = time.perf_counter() - t0
            elapsed_cpu = time.process_time() - c0
            print(f"wall={elapsed_wall:.6f}s cpu={elapsed_cpu:.6f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
