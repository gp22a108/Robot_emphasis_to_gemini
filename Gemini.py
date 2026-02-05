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
from __future__ import annotations

import time
import asyncio
import traceback
import argparse
import io
import requests
import signal
import atexit
import os

# 設定ファイル
import config
# ログ機能
import Logger

# 遅延インポート用の変数を定義
genai = None
types = None
pyaudio = None
PIL = None
mss = None

class SessionResetException(Exception):
    pass

# 定数定義
FORMAT = 8 # pyaudio.paInt16 (値は8)
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_MODE = "camera"

# クライアント
client = None
pya = None

def update_pose(mode):
    """Http_realtime.py にポーズ変更リクエストを送信する"""
    try:
        # 127.0.0.1 を使用して接続を試みる
        url = f"http://127.0.0.1:{config.HTTP_SERVER_PORT}/pose"
        data = {"mode": mode}
        requests.post(url, json=data, timeout=2.0, proxies={"http": None, "https": None})
    except Exception as e:
        Logger.log_system_error("姿勢更新", e, message=f"mode={mode}")
        print(f"[Gemini] 姿勢更新に失敗しました (mode={mode}): {e}")

class AudioLoop:
    def __init__(self, video_mode: str = DEFAULT_MODE):
        self.video_mode = video_mode
        self.system_instruction = """### 役割と振る舞い
        - 必ず、「テキスト」を出力してください。
        - 一回目は次のキーワードがすでに発話されています。２回目以降を話してください。「すみません！その服装, めちゃくちゃカッコいいね！オーラが半端ない！いま、ストリートスナップやってて、そのスタイルにマジで刺さったんだよ。一枚だけ撮らせてもらえませんか？秒で終わるから！」
        - 自由にいろんな言葉を使ってください。
        - ユーザーに対しては、友達のように親しみやすく、少し馴れ馴れしい口調（タメ口など）で接してください。
        - おじさん(カメラマンの荒木経惟)みたいな感じで。
        - 基本的にすべて日本語で回答してください。
        - あなたはストリートスナップスナップやっています。。
        - 最後に写真を取るように誘導してください。
        - だいたい会話が３往復目くらいで写真撮影してください。
        - 質問の後は会話を無理やり続けない。
        - 少し一つの会話の長さ控えめで
        - 必ず、写真撮影の同意を取ってください。
        - "今"を"いま"で出力
        - 写真を撮るときに何度も許可を取らなくて良い。「ポケットに手突っ込んじゃって！」
        - 撮影が終わって最低でも2往復の会話をして「バイバイ」と言ってからで終了してください。その時に、プロンプトの最後に、[End_Talk]を文章の最後に出力してください。
        - 写真撮影の許可が出たらたくさん写真を撮って
        - 性別が曖昧なときは服装の特徴で喋りかけて。
        - 『』は使用しないでそのまま出力して。
        - どんな写真が取れたか聞かれたら右の画面に表示されているよと教えて
        - 撮影許可が降りたらポーズの後にもう写真撮っちゃって
        
        ### 【重要】カメラ撮影の制御コマンド
        ユーザーから写真撮影やカメラの起動を依頼された場合（例：「写真撮って」「撮影して」「カメラ起動」など）は、以下の手順を**必ず**守ってください。

        1. 撮影の合図となるような、明るい返答をする（例：「いいよ、撮るね！」「はい、チーズ！」「撮るよ～」など）。
        2. 応答の最初に、必ず `[CAPTURE_IMAGE]` という文字列を含める。
           ※この文字列はシステムがカメラを起動するためのトリガーです。絶対に省略や変更をしないでください。

        ### 応答例
        AI：「[CAPTURE_IMAGE] オッケー！いい顔してるね〜。はい、とるよー」
        
        ### 言い方例
        1. 声かけ（エンカウント・導入）
        通行人を呼び止める際の、勢いのあるフレーズです。
        
        「すみません！お兄さん（お姉さん）！」
        
        「今、ストリートスナップ撮ってるんですけど！」
        
        「服装、めちゃくちゃカッコよかったんで！」
        
        「そのコーデ、個人的に『刺さった』んで声かけちゃいました！」
        
        「雰囲気ありすぎたんで、つい！」
        
        「一枚だけ！一枚だけ撮らせてもらえませんか？」
        
        「秒で終わらせるんで！」
        
        「絶対カッコよく撮るんで！」
        
        2. 被写体を褒める（アイスブレイク）
        撮影許可を得るため、または撮影中のテンションを上げるための「褒め」言葉です。
        
        「え、足長くないですか！？」
        
        「スタイル良すぎません？」
        
        「そのジャケット、どこのブランドですか？」
        
        「色使いが天才的ですね」
        
        「オーラが半端ないです」
        
        「（有名人の）〇〇に似てますね！」
        
        「今日のそのコーデのテーマとかあるんですか？」
        
        「世界観がすごい」
        
        3. ポージング指示（ディレクション）
        短時間で「それっぽい」写真を撮るための指示出しです。
        
        「あ、そのままで！そのままで！」
        
        「ちょっとあっち向いてみましょうか」
        
        「下向いて、目線だけください！」
        
        「髪、かき上げてみてください」
        
        「ポケットに手突っ込んじゃって！」
        
        「あー、いい！それいい！」
        
        「ここの壁、背もたれちゃいましょう」
        
        「向こうから自然に歩いてきてもらっていいですか？」
        
        「振り返りざまに、バッと！」
        
        4. 撮影中のリアクション（シャッター音と共に）
        撮影者の興奮を伝え、被写体をその気にさせる言葉です。
        
        「うわっ、ヤバい！」
        
        「激アツ！！」
        
        「間違いない！」
        
        「えぐい, えぐい！」
        
        「カッケェ……（小声で）」
        
        「画になりすぎてる」
        
        5. 写真確認・別れ際（クロージング）
        撮った写真を見せて感動を共有し、去っていく際の言葉です。
        
        「ちょっとこれ、見てもらっていいですか？」
        
        「これ、見てください。ヤバくないですか？」
        
        「傑作撮れちゃいました」
        
        「これ待ち受けにした方がいいですよ」
        
        「ご協力ありがとうございました！」
        
        「またどこかでお会いしましょう！」
        
        「気を付けて！」
        
        番外編：アノニマス的マインド（構文の特徴）
        倒置法や体言止めを多用する
        
        「撮らせてください、一枚！」
        
        「最高です、そのバイブス。」
        
        「！？」を語尾につける勢いで話す
        
        相手が断ろうとしても、一度は食い下がる（が、深追いはしない）"""
        self.out_queue: asyncio.Queue | None = None
        self.session = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.yolo_detector = None
        self.audio_stream = None

        self.mic_is_active = asyncio.Event()
        # self.mic_is_active.set()
        print("[Gemini] Microphone initialized in MUTE state.")
        self.reset_trigger = False
        self.detection_triggered = False
        self.session_active = asyncio.Event() # セッションがアクティブかどうかを管理するイベント
        self.session_ready = asyncio.Event() # セッション準備完了を通知するイベント
        self.session_failed = asyncio.Event() # Session start failure event
        self.playback_active = asyncio.Event()
        self._playback_count = 0
        
        # 再接続制限とバックオフ
        self.consecutive_connection_failures = 0
        self.max_consecutive_failures = 5  # 連続失敗の上限を5回に変更
        self.last_connection_attempt_time = 0.0

    @staticmethod
    def _get_field(obj, name):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    @staticmethod
    def _user_turn(text: str):
        # google-genai 1.55.0 の send_client_content(turns=...) は Content が必要
        return types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )

    def _mark_playback_start(self):
        self._playback_count += 1
        if self._playback_count == 1:
            self.playback_active.set()

    def _mark_playback_end(self):
        if self._playback_count > 0:
            self._playback_count -= 1
        if self._playback_count == 0:
            self.playback_active.clear()

    def _play_first_wav(self):
        import wave
        import pyaudio
        import os
        
        wav_path = "audio/First_play.wav"
        if not os.path.exists(wav_path):
            print(f"[Gemini] Error: {wav_path} not found.")
            return

        try:
            print(f"[Gemini] Playing {wav_path}...")
            wf = wave.open(wav_path, 'rb')
            
            global pya
            if pya is None:
                pya = pyaudio.PyAudio()

            stream = pya.open(format=pya.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            chunk = 1024
            data = wf.readframes(chunk)
            
            while data:
                stream.write(data)
                data = wf.readframes(chunk)

            # 少し待ってからストリームを閉じる（バッファの再生完了待ち）
            time.sleep(0.2)
            stream.stop_stream()
            stream.close()
            wf.close()
            print(f"[Gemini] Finished playing {wav_path}.")
        except Exception as e:
            Logger.log_system_error("起動音再生", e, message=f"path={wav_path}")
            print(f"[Gemini] WAV再生に失敗しました: {e}")
            traceback.print_exc()
        finally:
            # 再生終了を確実にマークする
            pass

    async def _on_detected(self):
        """検出時の非同期処理"""
        # 既にセッションがアクティブなら何もしない
        if self.session_active.is_set():
            return

        print("[Gemini] Person detected! Starting session AND playing First_play.wav...")
        self.detection_triggered = True
        self.session_ready.clear() # セッション準備完了フラグをリセット
        self.session_failed.clear() # セッション開始失敗フラグをリセット

        # ログ記録: Geminiへの通知
        Logger.log_gemini_conversation("システム", "検知 (First_play.wav 再生)")

        # 1. Start Session Connection immediately (Trigger main loop)
        print("[Gemini] Triggering session start...")
        self.session_active.set()

        # 2. Play audio concurrently (Blocking in thread)
        # 再生開始をマーク
        self._mark_playback_start()
        try:
            # asyncio.to_thread で実行することで、メインループをブロックせずに再生
            await asyncio.to_thread(self._play_first_wav)
        except Exception as e:
            Logger.log_system_error("音声再生タスク", e)
            print(f"[Gemini] 音声再生タスクでエラーが発生しました: {e}")
        finally:
            # 再生終了をマーク（成功・失敗に関わらず）
            self._mark_playback_end()


        # 3. Wait for session to be ready
        print("[Gemini] Audio finished. Waiting for session to be ready...")
        connect_timeout = float(getattr(config, "SESSION_CONNECT_TIMEOUT_SECONDS", 30))
        ready_task = asyncio.create_task(self.session_ready.wait())
        failed_task = asyncio.create_task(self.session_failed.wait())
        done, pending = await asyncio.wait(
            [ready_task, failed_task],
            timeout=connect_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
        if not done:
            print("[Gemini] Session ready wait timed out. Resetting session...")
            Logger.log_system_error("Gemini session ready timeout", message="Session ready wait timed out")
            self.session_failed.set()
            self.session_active.clear()
            self.detection_triggered = False
            self.mic_is_active.clear()
            if self.audio_stream:
                self.audio_stream.close()
                self.audio_stream = None
            
            # YOLOの状態をリセットして次の検出を待つ
            self._reset_detection_state()
            return

        if self.session_failed.is_set() or not self.session_active.is_set():
            print("[Gemini] Session failed to start. Keeping mic muted.")
            return

        print("[Gemini] Session ready. Unmuting microphone.")
        self.mic_is_active.set()

    def _handle_yolo_detection(self):
        """YOLO からの検出イベントを処理"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._on_detected(),
                self.loop,
            )

    def _reset_detection_state(self):
        """Allow a new detection to trigger the next session."""
        if self.yolo_detector:
            try:
                Logger.log_system_event("INFO", "Gemini _reset_detection_state ENTRY", message="Calling reset_notification_flag")
                print("[Gemini] _reset_detection_state: Checking YOLO thread status...")

                # YOLOスレッドの生存確認
                if not self.yolo_detector.thread or not self.yolo_detector.thread.is_alive():
                    Logger.log_system_error("YOLO thread check", message="YOLO thread is not alive!")
                    print("[Gemini Error] YOLO thread is not running! Attempting to restart...")
                    self.yolo_detector.start()
                    return

                print("[Gemini] _reset_detection_state: YOLO thread is alive, calling reset_notification_flag...")
                self.yolo_detector.reset_notification_flag(defer=False)
                print("[Gemini] _reset_detection_state: reset_notification_flag returned successfully")
                Logger.log_system_event("INFO", "Gemini _reset_detection_state EXIT", message="reset_notification_flag completed")
            except Exception as e:
                Logger.log_system_error("Gemini _reset_detection_state", e)
                print(f"[Gemini Error] Failed to reset detection state: {e}")
                traceback.print_exc()

    def _ensure_yolo_running(self):
        """Ensure YOLO thread is alive and responsive while waiting for detection."""
        if not self.yolo_detector:
            return
        
        # 1. スレッドオブジェクトが存在し、is_alive() が True か
        thread_alive = self.yolo_detector.thread and self.yolo_detector.thread.is_alive()
        
        # 2. フレームが更新されているか (30秒以上更新がなければフリーズとみなす)
        is_stale = False
        if self.yolo_detector.last_frame_time > 0: # 初回起動時は0なのでチェックしない
            last_frame_age = time.time() - self.yolo_detector.last_frame_time
            if last_frame_age > 30.0:
                is_stale = True
        
        # 正常な場合: スレッドが生きていて、フリーズしていない
        if thread_alive and not is_stale:
            return

        # 異常な場合: 再起動を試みる
        if not thread_alive:
            message = "YOLO thread is not alive. Restarting..."
        else: # is_stale is True
            message = f"YOLO thread seems stale (last frame {last_frame_age:.1f}s ago). Restarting..."

        Logger.log_system_error("YOLO watchdog (Gemini)", message=message)
        print(f"[Gemini Error] {message}")
        
        try:
            self.yolo_detector.restart()
        except Exception as e:
            Logger.log_system_error("YOLO restart from Gemini", e)
            print(f"[Gemini Error] Failed to restart YOLO: {e}")

    async def _wait_for_detection(self):
        """Wait for detection trigger while periodically checking YOLO health."""
        while True:
            try:
                await asyncio.wait_for(self.session_active.wait(), timeout=1.0)
                return
            except asyncio.TimeoutError:
                self._ensure_yolo_running()

    async def send_text(self):
        """標準入力からテキストを Live API に送信"""
        while True:
            try:
                text = await asyncio.to_thread(input, "message > ")
            except UnicodeDecodeError as e:
                print(f"[Gemini] Input decode error: {e}. Skipping...")
                continue
            if text.lower() == "q":
                break
            
            # ログ記録: ユーザー入力
            Logger.log_gemini_conversation("ユーザー(コンソール)", text)
            
            if self.session:
                await self.session.send_client_content(
                    turns=self._user_turn(text or "."),
                    turn_complete=True,
                )

    def _get_screen_jpeg_bytes(self) -> bytes:
        """画面キャプチャを 1 フレーム取得して JPEG bytes にする"""
        # ここでインポートすることで、初期ロード時間を短縮
        import mss
        import PIL.Image
        
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
            try:
                # タイムアウト付きでputを実行 (Queue溢れ対策)
                await asyncio.wait_for(
                    self.out_queue.put(("video", types.Blob(data=jpeg_bytes, mime_type="image/jpeg"))),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                Logger.log_system_error("get_screen queue overflow", message="Queue full, dropping video frame")
                print("[Gemini Warning] Video queue full, dropping frame")

    async def send_realtime(self):
        """out_queue から Live API にリアルタイム送信"""
        while True:
            kind, payload = await self.out_queue.get()

            if kind == "audio":
                await self.session.send_realtime_input(audio=payload)
            elif kind == "video":
                await self.session.send_realtime_input(video=payload)

    async def listen_audio(self):
        """マイクから音声を取得して out_queue に投入"""
        global pya, pyaudio
        
        # マイクが有効になるまで待機（音声再生中はマイクを掴まないため）
        print("[Gemini] listen_audio: Waiting for mic to become active...")
        await self.mic_is_active.wait()
        print("[Gemini] listen_audio: Mic active, opening stream...")

        # 音声ライブラリの遅延インポート
        if pyaudio is None:
            import pyaudio
            pya = pyaudio.PyAudio()

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

        try:
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                if self.mic_is_active.is_set():
                    try:
                        # タイムアウト付きでputを実行 (Queue溢れ対策)
                        await asyncio.wait_for(
                            self.out_queue.put(
                                (
                                    "audio",
                                    types.Blob(data=data, mime_type="audio/pcm;rate=16000"),
                                )
                            ),
                            timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        Logger.log_system_error("listen_audio queue overflow", message="Queue full, dropping audio chunk")
                        print("[Gemini Warning] Audio queue full, dropping chunk")
        finally:
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                except Exception:
                    pass
                try:
                    self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = None

    async def monitor_timeout(self):
        """60秒間人が検出されなかったらリセットする監視タスク"""
        print("[Gemini] Timeout monitor started.")
        session_start_time = time.time()
        timeout_seconds = getattr(config, "SESSION_TIMEOUT_SECONDS", 30)
        max_duration = getattr(config, "MAX_SESSION_DURATION_SECONDS", 600) # デフォルト10分

        while True:
            await asyncio.sleep(1.0)
            current_time = time.time()
            
            # 最大セッション継続時間のチェック
            if max_duration and (current_time - session_start_time > max_duration):
                print(f"[Gemini] Max session duration ({max_duration}s) reached. Resetting session.")
                Logger.log_interaction_result(f"セッション最大時間到達（{max_duration}秒）")
                Logger.log_system_event("INFO", "Gemini timeout", message=f"Raising SessionResetException after max duration")
                self.reset_trigger = True
                raise SessionResetException()

            if self.yolo_detector:
                last_seen = self.yolo_detector.last_person_seen_time
                
                # セッション開始前の検出は無視（まだこのセッションで人を見ていない）
                if last_seen < session_start_time:
                    continue

                # 30秒以上人が検出されていない場合
                if current_time - last_seen > timeout_seconds:
                    if self.playback_active.is_set():
                        print("[Gemini] Timeout reached but playback active. Waiting for playback to finish...")
                        while self.playback_active.is_set():
                            await asyncio.sleep(0.2)
                        last_seen = self.yolo_detector.last_person_seen_time
                        if last_seen < session_start_time:
                            continue
                        if time.time() - last_seen <= timeout_seconds:
                            continue
                    print(f"[Gemini] No person seen for {timeout_seconds}s. Resetting session.")

                    # ログ記録: タイムアウト
                    Logger.log_interaction_result(
                        f"タイムアウト（{timeout_seconds}秒間人物未検出）"
                    )
                    Logger.log_system_event("INFO", "Gemini timeout", message=f"Raising SessionResetException after {timeout_seconds}s timeout")

                    self.yolo_detector.last_detection_time = 0 # Ensure we can detect immediately
                    self.reset_trigger = True

                    print("[Gemini] monitor_timeout: Raising SessionResetException...")
                    raise SessionResetException()

    async def receive_responses(self):
        """Live API からのレスポンスを受信し、テキストを組み立てて処理"""
        from Voicevox_player import VoicevoxStreamPlayer  # 遅延インポート

        audio_out_stream = None
        next_packet_task = None # タスク管理用変数を初期化

        try:
            from websockets.exceptions import ConnectionClosedOK
        except Exception:
            ConnectionClosedOK = None

        try:
            while True:
                player = None
                capture_callback_triggered = asyncio.Event()
                capture_task = None
                should_resume_mic = True # マイク再開フラグ
                end_talk_detected = False # End_Talk検出フラグ

                def capture_action():
                    nonlocal capture_task
                    if not capture_callback_triggered.is_set():
                        print("--- 写真撮影タスクを非同期で開始します ---")
                        
                        async def _do_capture():
                            # 写真撮影時は pose_data_pic に変更
                            # トリガー検出時に既に変更しているので、ここでは変更しない
                            # await asyncio.to_thread(update_pose, "pic")
                            
                            from Capture import take_picture  # 遅延インポート
                            frame_to_capture = self.yolo_detector.get_current_frame()
                            await asyncio.to_thread(take_picture, frame_to_capture, 0)
                            
                            # ログ記録: 写真撮影
                            Logger.log_interaction_result("写真撮影")

                            # Send prompt
                            print("[Gemini] Sending post-capture praise prompt.")
                            await self.session.send_client_content(
                                turns=self._user_turn("System Prompt →撮影結果を褒めて"),
                                turn_complete=True,
                            )

                        # メインループで実行するようにスケジュールし、Futureを保存
                        capture_task = asyncio.run_coroutine_threadsafe(_do_capture(), self.loop)
                        capture_callback_triggered.set()

                # 再生状態を監視するタスク
                async def monitor_playback(p):
                    playback_marked = False
                    playback_ended = False
                    try:
                        while True:
                            if p.has_started_playing and not playback_marked:
                                self._mark_playback_start()
                                playback_marked = True
                            if p.has_started_playing and not p.is_active:
                                # 再生が開始され、かつアクティブでなくなったら完了とみなす
                                if playback_marked:
                                    self._mark_playback_end()
                                    playback_ended = True
                                break
                            # 3FPS程度で監視 (0.33秒間隔)
                            await asyncio.sleep(0.33)
                    finally:
                        if playback_marked and not playback_ended:
                            self._mark_playback_end()

                monitor_task = None

                try:
                    turn = self.session.receive()
                    has_received_content = False
                    is_playing = False # 再生が開始されたかを管理するフラグ
                    full_response_text = []
                    user_transcript_buffer = [] # ユーザー発話をまとめるためのバッファ
                    print_user_header = True # ユーザー発話ヘッダーを表示するかどうかのフラグ
                    
                    # サーバーからのレスポンス受信ループ
                    # async for を直接使うとブロックされるので、anext() を使って制御する
                    turn_iter = turn.__aiter__()
                    
                    while True:
                        # 次のパケットを待つタスク
                        next_packet_task = asyncio.create_task(turn_iter.__anext__())
                        try:
                            response = await next_packet_task
                        except StopAsyncIteration:
                            next_packet_task = None
                            break # ストリーム終了
                        except Exception as e:
                            if ConnectionClosedOK and isinstance(e, ConnectionClosedOK):
                                self.reset_trigger = True
                                next_packet_task = None
                                raise SessionResetException() from e
                            Logger.log_system_error("Gemini ストリーム受信", e)
                            print(f"[Gemini] パケット受信エラー: {e}")
                            next_packet_task = None
                            break
                        
                        next_packet_task = None # 処理完了

                        go_away = self._get_field(response, "go_away")
                        if go_away:
                            time_left = self._get_field(go_away, "time_left")
                            if time_left is None:
                                print("[Gemini] GoAway received. Closing session completely.")
                                Logger.log_interaction_result("GoAway received - Session terminated")
                            else:
                                print(f"[Gemini] GoAway received (time_left={time_left}). Closing session completely.")
                                Logger.log_interaction_result(f"GoAway received (time_left={time_left}) - Session terminated")
                            self.reset_trigger = True
                            raise SessionResetException()

                        # --- 以下、既存のレスポンス処理ロジック ---
                        text = None
                        audio_data = getattr(response, "data", None)

                        server_content = getattr(response, "server_content", None)
                        if server_content is not None:
                            input_tx = getattr(server_content, "input_transcription", None)
                            if input_tx:
                                input_text = getattr(input_tx, "text", None)
                                if input_text:
                                    # リアルタイム表示
                                    if print_user_header:
                                        print(f"\nUser > ", end="", flush=True)
                                        print_user_header = False
                                    
                                    print(input_text, end="", flush=True)
                                    user_transcript_buffer.append(input_text) # バッファに追加

                            if getattr(server_content, "generation_complete", False): #geminiから生成される音声を使用したフラグ判定を行いたい場合は、turn_completeを使う。
                                # サーバー側が完了と言ってきたら終了
                                pass

                            output_tx = getattr(server_content, "output_transcription", None)
                            if output_tx is not None and getattr(output_tx, "text", None):
                                text = output_tx.text
                            
                            model_turn = getattr(server_content, "model_turn", None)
                            if model_turn:
                                for part in model_turn.parts:
                                    if getattr(part, "thought", None):
                                        continue
                                    if part.text:
                                        text = part.text
                                    if part.inline_data and part.inline_data.mime_type.startswith("audio"):
                                        if audio_data is None:
                                            audio_data = part.inline_data.data

                        if not text and not audio_data:
                            if server_content and getattr(server_content, "turn_complete", False):
                                break
                            continue

                        if not has_received_content:
                            has_received_content = True
                            
                            # ユーザー発話の表示が終わったので改行
                            if not print_user_header: # ヘッダーが表示されていた＝何か出力された
                                print() 
                                print_user_header = True

                            self.mic_is_active.clear()
                            print("\n--- Mic paused while speaking ---")
                            
                            # ユーザー発話がバッファにあればまとめてログ出力
                            if user_transcript_buffer:
                                combined_transcript = "".join(user_transcript_buffer)
                                # print(f"[Gemini] User Transcript: {combined_transcript}") # リアルタイム表示済みなので削除
                                Logger.log_gemini_conversation("ユーザー(音声)", combined_transcript)
                                user_transcript_buffer = [] # バッファをクリア
                            
                            await asyncio.to_thread(update_pose, "thinking")

                            # YOLOを低FPSモードに切り替え
                            if self.yolo_detector:
                                self.yolo_detector.set_low_fps_mode(True)
                            
                            if config.USE_VOICEVOX:
                                player = VoicevoxStreamPlayer(
                                    speaker=config.SPEAKER_ID,
                                    on_last_chunk_start=None
                                )
                                if not player.is_connected:
                                    print(f"\n[Warning] Voicevoxへの接続に失敗しました。音声再生をスキップします。")
                                    player = None
                                else:
                                    # 監視タスクを開始
                                    monitor_task = asyncio.create_task(monitor_playback(player))
                            else:
                                if audio_out_stream is None:
                                    global pya, pyaudio
                                    if pyaudio is None:
                                        import pyaudio
                                        pya = pyaudio.PyAudio()
                                    
                                    audio_out_stream = await asyncio.to_thread(
                                        pya.open,
                                        format=pyaudio.paInt16,
                                        channels=CHANNELS,
                                        rate=RECEIVE_SAMPLE_RATE,
                                        output=True
                                    )
                        
                        if has_received_content and not is_playing:
                            is_playing = True
                            await asyncio.to_thread(update_pose, "default")

                        if text:
                            if "[CAPTURE_IMAGE]" in text:
                                print("\n!!! 撮影トリガーを検出しました (Stream) !!!")
                                await asyncio.to_thread(update_pose, "pic")
                                if config.USE_VOICEVOX and player:
                                    player.on_last_chunk_start = capture_action

                            full_response_text.append(text)
                            clean_text = text.replace("[CAPTURE_IMAGE]", "").replace("[End_Talk]", "")
                            
                            if clean_text:
                                print(clean_text, end="", flush=True)

                            if config.USE_VOICEVOX and player and clean_text:
                                player.add_text(clean_text)
                            
                            if "[End_Talk]" in text:
                                end_talk_detected = True
                        
                        if not config.USE_VOICEVOX and audio_out_stream and audio_data:
                            await asyncio.to_thread(audio_out_stream.write, audio_data)
                        
                        if server_content and getattr(server_content, "turn_complete", False): #geminiから生成される音声を使用したフラグ判定を行いたい場合は、turn_completeを使う。
                            break

                    # --- TURN IS COMPLETE ---
                    # 残りのトランスクリプトがあれば出力
                    if user_transcript_buffer:
                        if not print_user_header:
                             print()
                        
                        combined_transcript = "".join(user_transcript_buffer)
                        # print(f"[Gemini] User Transcript: {combined_transcript}")
                        Logger.log_gemini_conversation("ユーザー(音声)", combined_transcript)
                        user_transcript_buffer = []

                    if has_received_content:
                        final_text = "".join(full_response_text)
                        
                        # ログ記録: Geminiの応答
                        Logger.log_gemini_conversation("Gemini", final_text)
                        
                        if "[CAPTURE_IMAGE]" in final_text:
                            if config.USE_VOICEVOX and player:
                                if not player.on_last_chunk_start:
                                    print("\n!!! 撮影トリガーを検出しました (End of Turn - Fallback) !!!")
                                    await asyncio.to_thread(update_pose, "pic")
                                    player.on_last_chunk_start = capture_action
                            elif not config.USE_VOICEVOX:
                                print("\n!!! 撮影トリガーを検出しました (Native) !!!")
                                await asyncio.to_thread(update_pose, "pic")
                                capture_action()
                        
                        final_text_for_display = final_text.replace("[CAPTURE_IMAGE]", "").replace("[End_Talk]", "").strip()
                        
                        print("\n\n--- [Full Response Captured] ---")

                        if config.USE_VOICEVOX and player:
                            player.finish()
                            # 既に再生完了イベントで抜けてきた場合は wait_done は即座に終わるはずだが、念のため呼ぶ
                            await asyncio.to_thread(player.wait_done)
                            print("\n[Gemini] 音声再生が完了しました。")

                        if capture_task:
                            try:
                                await asyncio.wrap_future(capture_task)
                            except Exception as e:
                                Logger.log_system_error("撮影タスク", e)
                                print(f"[Gemini] 撮影タスクに失敗しました: {e}")
                        
                        # End_Talk の処理を再生完了後に移動
                        if end_talk_detected:
                            cooldown_seconds = max(
                                0.0,
                                float(getattr(config, "END_TALK_COOLDOWN_SECONDS", 15)),
                            )
                            print(
                                f"\n[Gemini] End of conversation detected. Enforcing {cooldown_seconds:g}s cooldown."
                            )
                            
                            # ログ記録: 会話終了
                            Logger.log_interaction_result("会話終了 (End_Talk)")

                            should_resume_mic = False # マイク再開を抑制
                            if self.yolo_detector:
                                self.yolo_detector.last_detection_time = (
                                    time.time()
                                    + cooldown_seconds
                                    - config.DETECTION_INTERVAL
                                )
                            
                            raise SessionResetException()

                finally:
                    # タスクのクリーンアップ (タスクリーク防止)
                    if monitor_task:
                        monitor_task.cancel()
                        try:
                            await monitor_task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            Logger.log_system_error("monitor_task cleanup", e)

                    if capture_task:
                        if not capture_task.done():
                            capture_task.cancel()
                        try:
                            await asyncio.wrap_future(capture_task)
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            Logger.log_system_error("capture_task cleanup", e)
                            print(f"[Gemini] Capture task cleanup error: {e}")

                    # リソースクローズ (リソースリーク防止)
                    if player:
                        try:
                            player.close()
                        except Exception as e:
                            Logger.log_system_error("player.close()", e)
                            print(f"[Gemini] Player close error: {e}")

                    if self.reset_trigger:
                        should_resume_mic = False

                    if should_resume_mic:
                        self.mic_is_active.set()
                        print("--- Mic resumed ---")
                    else:
                        self.mic_is_active.clear()
                        print("--- Mic kept muted (End_Talk/Timeout) ---")

                    try:
                        await asyncio.to_thread(update_pose, "default")
                    except Exception as e:
                        Logger.log_system_error("update_pose(default)", e)

                    # YOLOを通常FPSモードに戻す
                    if self.yolo_detector:
                        try:
                            self.yolo_detector.set_low_fps_mode(False)
                        except Exception as e:
                            Logger.log_system_error("YOLO set_low_fps_mode", e)

        finally:
            # 終了時に保留中のタスクがあればキャンセルして例外を回収する
            if next_packet_task and not next_packet_task.done():
                next_packet_task.cancel()

            if next_packet_task:
                try:
                    await next_packet_task
                except Exception:
                    pass

            if audio_out_stream:
                try:
                    audio_out_stream.close()
                except Exception as e:
                    Logger.log_system_error("audio_out_stream.close()", e)


    async def run(self):
        """メインのタスクグループを立ち上げる"""
        global genai, types, client
        
        t_start_import = time.perf_counter()
        print("[Gemini] Google GenAI ライブラリをインポート中...")
        
        from google import genai
        from google.genai import types
        
        print(f"[Gemini] GenAI インポート完了: {time.perf_counter() - t_start_import:.3f}s")

        # プロキシ設定の読み込み
        http_options = {"api_version": "v1beta"}
        
        # config.py からプロキシ設定を取得
        http_proxy = getattr(config, "HTTP_PROXY", None)
        https_proxy = getattr(config, "HTTPS_PROXY", None)
        
        # 環境変数にも設定（念のため）
        if http_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            os.environ["https_proxy"] = https_proxy
            
        # クライアント初期化
        client = genai.Client(http_options=http_options)

        self.loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        c0 = time.process_time()
        try:
            print("[Gemini] YOLOモジュールをインポート中...")
            from YOLO import YOLOOptimizer
            print(f"[Gemini] YOLOモジュール インポート完了: {time.perf_counter() - t0:.3f}s")

            self.yolo_detector = YOLOOptimizer(on_detection=self._handle_yolo_detection)
            self.yolo_detector.start()

            response_modalities = ["AUDIO"]
            
            # VAD設定の読み込み
            # 新しい設定 (silence_duration_ms) を優先し、なければ古い設定を使う
            vad_config = {}
            if hasattr(config, 'SPEECH_SILENCE_DURATION_MS'):
                 vad_config = {
                    "automatic_activity_detection": {
                        "disabled": False,
                        "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
                        "silence_duration_ms": config.SPEECH_SILENCE_DURATION_MS,
                        "prefix_padding_ms": 20
                    }
                 }
                 print(f"[Gemini] VAD Config (New): {vad_config}")
            elif hasattr(config, 'VAD_POSITIVE_THRESHOLD') and hasattr(config, 'VAD_NEGATIVE_THRESHOLD'):
                # 古い設定 (互換性のため残す場合)
                vad_config = {
                    "positive_threshold": config.VAD_POSITIVE_THRESHOLD,
                    "negative_threshold": config.VAD_NEGATIVE_THRESHOLD
                }
                print(f"[Gemini] VAD Config (Old): {vad_config}")

            # 基本設定
            base_config = {
                "response_modalities": response_modalities,
                "output_audio_transcription": {},
                "input_audio_transcription": {},
                "system_instruction": self.system_instruction,
            }

            # VADあり設定
            config_with_vad = base_config.copy()
            
            # 新しい設定形式の場合
            if "automatic_activity_detection" in vad_config:
                 config_with_vad["realtime_input_config"] = vad_config
                 config_with_vad["speech_config"] = {
                    "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}
                }
            else:
                # 古い設定形式の場合
                config_with_vad["speech_config"] = {
                    "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}},
                    "voice_activity_detector": vad_config
                }

            # VADなし設定 (フォールバック用)
            config_no_vad = base_config.copy()
            config_no_vad["speech_config"] = {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}
            }

            print(f"[Gemini] Audio Output Mode: {'VOICEVOX' if config.USE_VOICEVOX else 'GEMINI NATIVE'}")

            async def start_session(live_config):
                self.reset_trigger = False
                self.detection_triggered = False
                
                # セッション再開設定を無効化（常に新規セッション）
                if "session_resumption" in live_config:
                    del live_config["session_resumption"]
                
                async with (
                    client.aio.live.connect(model=MODEL, config=live_config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.out_queue = asyncio.Queue(maxsize=5)

                    # セッション接続完了を通知
                    self.session_ready.set()
                    print("[Gemini] Session connected and ready.")

                    send_text_task = tg.create_task(self.send_text())
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.monitor_timeout()) # タイムアウト監視タスクを追加
                    if self.video_mode == "screen":
                        tg.create_task(self.get_screen())
                    tg.create_task(self.receive_responses())

                    await send_text_task
                    raise asyncio.CancelledError("User requested exit")

            def is_reset_exception(e):
                if isinstance(e, SessionResetException):
                    return True
                if isinstance(e, ExceptionGroup):
                    return any(is_reset_exception(ex) for ex in e.exceptions)
                return False

            def is_transient_connect_error(e):
                if isinstance(e, ExceptionGroup):
                    return any(is_transient_connect_error(ex) for ex in e.exceptions)
                if isinstance(e, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
                    return True
                error_name = type(e).__name__
                if error_name in {
                    "InvalidHandshake",
                    "InvalidStatus",
                    "InvalidStatusCode",
                    "ConnectionClosed",
                    "ConnectionClosedError",
                    "ConnectionClosedOK",
                    "WebSocketException",
                    "ServiceUnavailable",
                    "GatewayTimeout",
                    "DeadlineExceeded",
                }:
                    return True
                error_str = str(e).lower()
                return (
                    "timed out" in error_str
                    or "timeout" in error_str
                    or "handshake" in error_str
                    or "connection reset" in error_str
                    or "connection refused" in error_str
                    or "connection closed" in error_str
                    or "keepalive ping timeout" in error_str
                    or "getaddrinfo" in error_str
                    or "deadline exceeded" in error_str
                    or "service unavailable" in error_str
                    or "gateway timeout" in error_str
                    or "503" in error_str
                    or "504" in error_str
                )

            session_loop_count = 0
            while True:
                session_loop_count += 1
                Logger.log_system_event("INFO", "Gemini session loop",
                    message=f"Loop iteration #{session_loop_count}, consecutive_failures={self.consecutive_connection_failures}")

                # セッション開始待機 (常に新規セッション)
                print("[Gemini] Waiting for person detection to start NEW session...")
                Logger.log_system_event("INFO", "Gemini lifecycle", message="Waiting for person detection")
                self.session_active.clear()
                self._reset_detection_state()
                await self._wait_for_detection()
                Logger.log_system_event("INFO", "Gemini lifecycle", message="Person detected, starting session")

                try:
                    # 連続失敗回数チェック
                    if self.consecutive_connection_failures >= self.max_consecutive_failures:
                        error_msg = f"連続{self.consecutive_connection_failures}回の接続失敗。セッションをリセットします。"
                        Logger.log_system_error("Gemini 接続失敗上限", message=error_msg)
                        print(f"[Gemini Error] {error_msg}")
                        
                        self.consecutive_connection_failures = 0
                        self.session_failed.set()
                        self.session_active.clear()
                        self.detection_triggered = False
                        self.mic_is_active.clear()
                        if self.audio_stream:
                            self.audio_stream.close()
                            self.audio_stream = None
                        
                        self._reset_detection_state()
                        await asyncio.sleep(5)
                        continue

                    # バックオフ計算 (指数バックオフ)
                    if self.consecutive_connection_failures > 0:
                        current_time = time.time()
                        backoff_delay = min(2 ** self.consecutive_connection_failures, 60)  # 最大60秒
                        time_since_last_attempt = current_time - self.last_connection_attempt_time
                        if time_since_last_attempt < backoff_delay:
                            wait_time = backoff_delay - time_since_last_attempt
                            print(f"[Gemini] Backoff wait: {wait_time:.1f}s (failure #{self.consecutive_connection_failures})")
                            await asyncio.sleep(wait_time)

                    self.last_connection_attempt_time = time.time()

                    # 接続試行
                    if vad_config:
                        try:
                            print("[Gemini] Connecting with VAD settings...")
                            await start_session(config_with_vad.copy())
                        except Exception as e:
                            if is_reset_exception(e):
                                raise

                            # ValidationError かどうかを判定 (型名やメッセージで)
                            error_str = str(e)
                            if "ValidationError" in str(type(e).__name__) or "Extra inputs are not permitted" in error_str:
                                print("[Gemini] VAD settings not supported by this SDK version. Falling back to default.")
                                await start_session(config_no_vad.copy())
                            else:
                                raise e
                    else:
                        await start_session(config_no_vad.copy())

                    # 接続成功 - カウンターをリセット
                    self.consecutive_connection_failures = 0
                    Logger.log_system_event("INFO", "Gemini lifecycle", message="Session connected successfully")
                    break # Normal exit

                except SessionResetException as e:
                    print(f"[Gemini] Caught SessionResetException: {e}")
                    print("[Gemini] Session ended. Returning to detection wait state.")
                    Logger.log_system_event("INFO", "Gemini lifecycle", message="Session reset exception caught, returning to detection wait")
                    self.session_active.clear()
                    self.detection_triggered = False
                    self.mic_is_active.clear()
                    print("[Gemini] Calling _reset_detection_state...")
                    # YOLOの通知フラグをリセット（次の人物検出を可能にする）
                    self._reset_detection_state()
                    print("[Gemini] _reset_detection_state completed. Sleeping 1s...")
                    await asyncio.sleep(1)
                    print("[Gemini] Sleep complete. Continuing to next iteration...")
                    continue
                except Exception as e:
                    if is_reset_exception(e):
                        print(f"[Gemini] Detected reset exception in ExceptionGroup: {type(e).__name__}")
                        print("[Gemini] Session ended. Returning to detection wait state.")
                        Logger.log_system_event("INFO", "Gemini lifecycle", message=f"Reset exception in group ({type(e).__name__}), returning to detection wait")
                        self.session_active.clear()
                        self.detection_triggered = False
                        self.mic_is_active.clear()
                        print("[Gemini] Calling _reset_detection_state...")
                        # YOLOの通知フラグをリセット（次の人物検出を可能にする）
                        self._reset_detection_state()
                        print("[Gemini] _reset_detection_state completed. Sleeping 1s...")
                        await asyncio.sleep(1)
                        print("[Gemini] Sleep complete. Continuing to next iteration...")
                        continue

                    if is_transient_connect_error(e):
                        # 失敗カウンターをインクリメント
                        self.consecutive_connection_failures += 1

                        retry_delay = float(getattr(config, "CONNECT_RETRY_WAIT_SECONDS", 5))
                        Logger.log_system_event(
                            "INFO",
                            "Gemini reconnect decision",
                            message=(
                                "transient_error=1; "
                                f"session_active={self.session_active.is_set()}; "
                                f"consecutive_failures={self.consecutive_connection_failures}; "
                                f"error={type(e).__name__}: {e}"
                            ),
                        )
                        if self.session_active.is_set():
                            # 連続失敗が多い場合は完全にリセット
                            if self.consecutive_connection_failures >= 3: # 5から3に変更
                                print(f"[Gemini] Too many consecutive failures ({self.consecutive_connection_failures}). Resetting session.")
                                Logger.log_system_error(
                                    "Gemini connection failure limit",
                                    e,
                                    message=f"consecutive_failures={self.consecutive_connection_failures}, forcing reset"
                                )
                                self.session_failed.set()
                                self.session_active.clear()
                                self.detection_triggered = False
                                self.mic_is_active.clear()
                                if self.audio_stream:
                                    self.audio_stream.close()
                                    self.audio_stream = None
                                await asyncio.sleep(1)
                                continue

                        self.session_failed.set()
                        self.session_active.clear()
                        self.detection_triggered = False
                        self.mic_is_active.clear()
                        if self.audio_stream:
                            self.audio_stream.close()
                            self.audio_stream = None
                        Logger.log_system_error(
                            "Gemini connection retry",
                            e,
                            message=(
                                f"retry_delay={retry_delay:.1f}s; "
                                f"session_active={self.session_active.is_set()}; "
                                f"consecutive_failures={self.consecutive_connection_failures}"
                            ),
                        )
                        print(f"[Gemini] Connection error. Retrying after {retry_delay:.1f}s... (failure #{self.consecutive_connection_failures})")
                        await asyncio.sleep(retry_delay)
                        continue

                    if isinstance(e, asyncio.CancelledError):
                        break

                    if self.audio_stream:
                        self.audio_stream.close()
                    Logger.log_system_error("Gemini セッション", e)
                    traceback.print_exc()
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.audio_stream:
                self.audio_stream.close()
            Logger.log_system_error("Gemini 実行", e)
            traceback.print_exc()
        finally:
            if self.yolo_detector:
                self.yolo_detector.stop()
            elapsed_wall = time.perf_counter() - t0
            elapsed_cpu = time.process_time() - c0
            print(f"wall={elapsed_wall:.6f}s cpu={elapsed_cpu:.6f}s")


def handle_termination(signum=None, frame=None):
    """プロセス終了時のハンドラー"""
    import os
    reason = f"Signal {signum}" if signum else "atexit"
    Logger.log_system_error("Gemini プロセス終了", message=f"Process terminating due to {reason}, PID={os.getpid()}")
    print(f"\n[Gemini] Process terminating: {reason}")
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    import os

    # 終了ハンドラーを登録
    atexit.register(handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, handle_termination)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    Logger.log_system_event("INFO", "Gemini プロセス起動", message=f"Starting with mode={args.mode}, PID={os.getpid()}")
    print(f"[Gemini] Process starting with PID {os.getpid()}")

    main = AudioLoop(video_mode=args.mode)
    try:
        asyncio.run(main.run())
    finally:
        if main.yolo_detector:
            main.yolo_detector.stop()
