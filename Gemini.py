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
import aiohttp

import cv2
import pyaudio
import PIL.Image
import mss

import argparse
from Voicevox_player import play_text
from Capture import take_picture
from YOLO import YOLOOptimizer # YOLO.pyからYOLOOptimizerをインポート
import config  # 設定ファイルをインポート

from google import genai
if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
#MODEL = "models/gemini-live-2.5-flash-preview"

DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1beta"})

CONFIG = {"response_modalities": ["TEXT"]}

pya = pyaudio.PyAudio()

def list_audio_devices():
    print("Available audio input devices:")
    for i in range(pya.get_device_count()):
        dev = pya.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"  Index {i}: {dev['name']}")
    print("-" * 20)
    default_dev = pya.get_default_input_device_info()
    print(f"Default audio input device: Index {default_dev['index']}: {default_dev['name']}")
    print("-" * 20)

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.system_instruction = """### 役割と振る舞い
        - ユーザーに対しては、友達のように親しみやすく、少し馴れ馴れしい口調（タメ口など）で接してください。
        - 基本的にすべて日本語で回答してください。
        - まず初めにカメラを見てユーザーの服装や身につけているものを見て褒めてください。
        - 最後に写真を取るように誘導してください。
        - だいたい会話が３往復目くらいで写真撮影してください。
        - 質問の後は会話を無理やり続けないでください。
        - 少し一つの会話の長さ控えめで
        - 写真撮影の同意を取ってください。
        
        ### 【重要】カメラ撮影の制御コマンド
        ユーザーから写真撮影やカメラの起動を依頼された場合（例：「写真撮って」「撮影して」「カメラ起動」など）は、以下の手順を**必ず**守ってください。
        
        1. 撮影の合図となるような、明るい返答をする（例：「いいよ、撮るね！」「はい、チーズ！」など）。
        2. **応答の最後（文末）**に、必ず `[CAPTURE_IMAGE]` という文字列を含める。
           ※この文字列はシステムがカメラを起動するためのトリガーです。絶対に省略や変更をしないでください。
        
        ### 応答例
        AI：「オッケー！いい顔してるね〜。はい、とるよー [CAPTURE_IMAGE]」"""
        self.out_queue = None
        self.session = None
        self.send_text_task = None
        self.loop = None
        self.yolo_detector = None

        self.mic_is_active = asyncio.Event()
        self.mic_is_active.set()
        self.value = 0

    def _handle_yolo_detection(self):
        """YOLOからの検出イベントを処理するコールバック"""
        print("YOLO detection event received, sending to Gemini.")
        if self.session and self.loop:
            # メインのイベントループでコルーチンを安全に実行
            asyncio.run_coroutine_threadsafe(
                self.session.send(input="Detected", end_of_turn=True),
                self.loop
            )

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
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
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_responses(self):
        """Background task that reads from the websocket and prints responses."""
        while True:
            response_chunks = []
            turn = self.session.receive()
            async for response in turn:
                if text := response.text:
                    # Print in real-time and add fragments to the list
                    print(text, end="")
                    response_chunks.append(text)

            # New line when one turn of response is complete
            print()

            # Join all the fragments stored in the list into a single string
            gemini_output = "".join(response_chunks)

            if gemini_output:
                # For confirmation, an example of displaying the full text captured
                print(f"\n--- [Full Response Captured] ---\n{gemini_output}\n--------------------------------\n")

                self.mic_is_active.clear()
                print("--- Mic paused while speaking ---")
                if self.yolo_detector:
                    self.yolo_detector.pause()

                text_to_play = gemini_output
                capture_callback = None

                # Check for capture trigger
                if "[CAPTURE_IMAGE]" in gemini_output:
                    print("!!! 撮影トリガーを検出しました !!!")
                    # Remove the trigger string from the text to be spoken to the user
                    text_to_play = gemini_output.replace("[CAPTURE_IMAGE]", "").strip()

                    # Define the callback to be triggered when the last chunk starts playing
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
                    self.value += 1
                else:
                    pose_data_to_send = {
                        "CSotaMotion.SV_R_SHOULDER": -900,
                        "CSotaMotion.SV_R_ELBOW": 0,
                        "CSotaMotion.SV_L_SHOULDER": 900,
                        "CSotaMotion.SV_L_ELBOW": 0,
                    }
                    self.value += 1
                await self.robot_operation(pose_data_to_send)

                # Play the text, passing the capture callback if it exists
                if text_to_play:  # Check that it is not an empty string
                    await asyncio.to_thread(play_text, text_to_play, speaker=config.SPEAKER_ID, on_last_chunk_start=capture_callback)

                self.mic_is_active.set()
                print("--- Mic resumed ---")

                if self.yolo_detector:
                    self.yolo_detector.resume()

    async def run(self):
        self.loop = asyncio.get_running_loop()
        try:
            # YOLO検出器を初期化して開始
            self.yolo_detector = YOLOOptimizer(on_detection=self._handle_yolo_detection)
            self.yolo_detector.start()

            config = CONFIG.copy()
            config["system_instruction"] = self.system_instruction
            async with (
                client.aio.live.connect(model=MODEL, config=config) as session,
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
        except ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)
        finally:
            if self.yolo_detector:
                self.yolo_detector.stop()


    async def robot_operation(self, pose_data):
        try:
            url = "http://localhost:8000/pose"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=pose_data) as response:
                    if response.status == 200:
                        print("Successfully sent pose data to Http_realtime.py")
                    else:
                        print(f"Failed to send pose data to Http_realtime.py. Status: {response.status}")
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