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

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss aiohttp
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones.

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""
import asyncio
import base64
import io
import sys
import traceback
import argparse

import aiohttp
import cv2
import pyaudio
import PIL.Image
import mss

from Voicevox_player import play_text
from Capture import take_picture
from YOLO import YOLOOptimizer  # YOLO.py から YOLOOptimizer をインポート
import config  # 設定ファイル

from google import genai

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
        self.system_instruction = """### 役割と振る舞いc
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
        self.yolo_detector: YOLOOptimizer | None = None
        self.audio_stream = None

        self.mic_is_active = asyncio.Event()
        self.mic_is_active.set()

        self.value = 0  # ロボットポーズの切り替え用

    def _handle_yolo_detection(self):
        """YOLO からの検出イベントを処理"""
        print("YOLO detection event received, sending to Gemini.")
        if self.session and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.session.send(input="Detected", end_of_turn=True),
                self.loop,
            )

    async def send_text(self):
        """標準入力からテキストを Live API に送信"""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_screen(self):
        """画面キャプチャを 1 フレーム取得"""
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            grabbed = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(grabbed.rgb, grabbed.size)

        img = PIL.Image.open(io.BytesIO(image_bytes))
        buf = io.BytesIO()
        img.save(buf, format="jpeg")
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode()
        return {"mime_type": mime_type, "data": encoded}

    async def get_screen(self):
        """画面キャプチャを連続で Live API に送出"""
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        """out_queue から Live API にリアルタイム送信"""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        """マイクから音声を取得して Live API に送信"""
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
            data = await asyncio.to_thread(
                self.audio_stream.read, CHUNK_SIZE, **kwargs
            )
            if self.mic_is_active.is_set():
                await self.out_queue.put(
                    {
                        "data": data,
                        "mime_type": "audio/pcm;rate=16000",
                    }
                )

    async def receive_responses(self):
        """Live API からのレスポンスを受信し、テキストを組み立てて処理"""
        while True:
            response_chunks: list[str] = []

            # 1 ターン分のレスポンスストリーム
            turn = self.session.receive()

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

                print(text, end="")
                response_chunks.append(text)

            print()
            gemini_output = "".join(response_chunks)

            if gemini_output:
                print("\n--- [Full Response Captured] ---")
                print(gemini_output)
                print("--------------------------------\n")

            # モデル発話中はマイク／YOLO を一時停止
            self.mic_is_active.clear()
            print("--- Mic paused while speaking ---")
            if self.yolo_detector:
                self.yolo_detector.pause()

            text_to_play = gemini_output
            capture_callback = None

            # 撮影トリガー検出
            if "[CAPTURE_IMAGE]" in gemini_output:
                print("!!! 撮影トリガーを検出しました !!!")
                text_to_play = gemini_output.replace("[CAPTURE_IMAGE]", "").strip()

                def capture_action():
                    print("--- 写真撮影タスクを非同期で開始します ---")
                    frame_to_capture = self.yolo_detector.get_current_frame()
                    coro = asyncio.to_thread(take_picture, frame_to_capture, 0)
                    asyncio.run_coroutine_threadsafe(coro, self.loop)

                capture_callback = capture_action

            # --- Robot motion ---
            if self.value % 2 == 0:
                pose_data_to_send = {
                    "CSotaMotion.SV_R_SHOULDER": 800,
                    "CSotaMotion.SV_R_ELBOW": 0,
                    "CSotaMotion.SV_L_SHOULDER": -800,
                    "CSotaMotion.SV_L_ELBOW": 0,
                }
            else:
                pose_data_to_send = {
                    "CSotaMotion.SV_R_SHOULDER": -900,
                    "CSotaMotion.SV_R_ELBOW": 0,
                    "CSotaMotion.SV_L_SHOULDER": 900,
                    "CSotaMotion.SV_L_ELBOW": 0,
                }
            self.value += 1

            await self.robot_operation(pose_data_to_send)

            # Voicevox で読み上げ
            if text_to_play:
                await asyncio.to_thread(
                    play_text,
                    text_to_play,
                    speaker=config.SPEAKER_ID,
                    on_last_chunk_start=capture_callback,
                )

            # マイク／YOLO を再開
            self.mic_is_active.set()
            print("--- Mic resumed ---")
            if self.yolo_detector:
                self.yolo_detector.resume()

    async def run(self):
        """メインのタスクグループを立ち上げる"""
        self.loop = asyncio.get_running_loop()
        try:
            # YOLO 検出器を初期化して開始
            self.yolo_detector = YOLOOptimizer(
                on_detection=self._handle_yolo_detection
            )
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

    async def robot_operation(self, pose_data: dict):
        """ロボット用 HTTP API にポーズデータを送信"""
        try:
            url = "http://localhost:8000/pose"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=pose_data) as response:
                    if response.status == 200:
                        print("Successfully sent pose data to Http_realtime.py")
                    else:
                        print(
                            f"Failed to send pose data to Http_realtime.py. "
                            f"Status: {response.status}"
                        )
        except Exception as e:
            print(f"Failed to send pose data to Http_realtime.py: {e}")


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
