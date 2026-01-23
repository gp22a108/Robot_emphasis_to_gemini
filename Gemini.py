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
import threading

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
        requests.post(url, json=data, timeout=2.0)
    except Exception as e:
        print(f"Failed to update pose to {mode}: {e}")

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

        1. 応答の最初に、必ず `[CAPTURE_IMAGE]` という文字列を含める。
        2. 次に、撮影の合図となるような掛け声を言う（例：「いいよ、撮るね！」「はい、チーズ！」「撮るよ～」など）。
        3. **重要**: 掛け声の直後に必ず「改行」を入れてください。システムは改行のタイミングで写真を撮影します。
        4. 改行の後に、撮影後の感想やリアクション（例：「撮れた！」「最高！」など）を続けてください。

        ### 応答例
        AI：「[CAPTURE_IMAGE] オッケー！いい顔してるね〜。はい、とるよー
        （ここでシステムが撮影）
        うわ、めっちゃいい写真撮れた！最高！」
        
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
        
        「えぐい、えぐい！」
        
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

    @staticmethod
    def _user_turn(text: str):
        # google-genai 1.55.0 の send_client_content(turns=...) は Content が必要
        return types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )

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
            print(f"[Gemini] Error playing wav: {e}")
            traceback.print_exc()

    async def _on_detected(self):
        """検出時の非同期処理"""
        # 既にセッションがアクティブなら何もしない
        if self.session_active.is_set():
            return
            
        print("[Gemini] Person detected! Starting session AND playing First_play.wav...")
        self.detection_triggered = True
        
        # ログ記録: Geminiへの通知
        Logger.log_gemini_conversation("System", "Detected (Playing First_play.wav)")
        
        # 1. Start Session Connection immediately (Trigger main loop)
        print("[Gemini] Triggering session start...")
        self.session_active.set() 
        
        # 2. Play audio concurrently (Blocking in thread)
        await asyncio.to_thread(self._play_first_wav)
        
        print("[Gemini] Audio finished. Unmuting microphone.")
        self.mic_is_active.set()

    def _handle_yolo_detection(self):
        """YOLO からの検出イベントを処理"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._on_detected(),
                self.loop,
            )

    async def send_text(self):
        """標準入力からテキストを Live API に送信"""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            
            # ログ記録: ユーザー入力
            Logger.log_gemini_conversation("User (Console)", text)
            
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
            await self.out_queue.put(
                ("video", types.Blob(data=jpeg_bytes, mime_type="image/jpeg"))
            )

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

        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if self.mic_is_active.is_set():
                await self.out_queue.put(
                    (
                        "audio",
                        types.Blob(data=data, mime_type="audio/pcm;rate=16000"),
                    )
                )

    async def monitor_timeout(self):
        """人が検出されなかったらリセットする監視タスク"""
        print("[Gemini] Timeout monitor started.")
        session_start_time = time.time()
        while True:
            await asyncio.sleep(1.0)
            if self.yolo_detector:
                last_seen = self.yolo_detector.last_person_seen_time
                
                # セッション開始前の検出は無視（まだこのセッションで人を見ていない）
                if last_seen < session_start_time:
                    continue

                # 設定された時間以上人が検出されていない場合
                if time.time() - last_seen > config.SESSION_TIMEOUT_SECONDS:
                    print(f"[Gemini] No person seen for {config.SESSION_TIMEOUT_SECONDS}s. Resetting session.")
                    
                    # ログ記録: タイムアウト
                    Logger.log_interaction_result(f"Timeout (No person seen for {config.SESSION_TIMEOUT_SECONDS}s)")
                    
                    self.yolo_detector.last_detection_time = 0 # Ensure we can detect immediately
                    self.reset_trigger = True
                    raise SessionResetException()

    async def receive_responses(self):
        """Live API からのレスポンスを受信し、テキストを組み立てて処理"""
        from Voicevox_player import VoicevoxStreamPlayer  # 遅延インポート

        audio_out_stream = None
        next_packet_task = None # タスク管理用変数を初期化
        response_timer_task = None # 応答待ちタイマー

        try:
            while True:
                player = None
                playback_finished_event = asyncio.Event() # 再生完了イベント
                should_resume_mic = True # マイク再開フラグ
                end_talk_detected = False # End_Talk検出フラグ
                capture_pending = False # [CAPTURE_IMAGE]検出済みだが未撮影フラグ

                # 撮影実行用関数（非同期）
                async def perform_capture():
                    print("\n--- 写真撮影を実行します ---")
                    try:
                        # 写真撮影時は pose_data_pic に変更
                        # await asyncio.to_thread(update_pose, "pic")
                        
                        from Capture import take_picture  # 遅延インポート
                        frame_to_capture = self.yolo_detector.get_current_frame()
                        # 0秒待機で撮影
                        await asyncio.to_thread(take_picture, frame_to_capture, 0)
                        
                        # ログ記録: 写真撮影
                        Logger.log_interaction_result("Picture Taken")
                        
                        # 撮影後はデフォルトポーズに戻す
                        await asyncio.to_thread(update_pose, "default")
                    except Exception as e:
                        print(f"Capture failed: {e}")
                        traceback.print_exc()

                # Voicevox用コールバック（同期）
                def capture_callback():
                    # Voicevoxスレッドから呼ばれることを想定
                    if threading.current_thread() is threading.main_thread():
                        # メインスレッドなら非同期タスクとして投げるだけ（待たない）
                        asyncio.run_coroutine_threadsafe(perform_capture(), self.loop)
                        return

                    print("--- Voicevox callback: Waiting for capture... ---")
                    future = asyncio.run_coroutine_threadsafe(perform_capture(), self.loop)
                    try:
                        future.result() # 完了までブロック
                        print("--- Voicevox callback: Capture finished ---")
                    except Exception as e:
                        print(f"Capture callback failed: {e}")

                # 再生状態を監視するタスク
                async def monitor_playback(p):
                    started = False
                    while True:
                        if not started and p.has_started_playing:
                            started = True
                            # 撮影待機中でなければ default に戻す
                            if not capture_pending:
                                await asyncio.to_thread(update_pose, "default")

                        if p.has_started_playing and not p.is_active:
                            # 再生が開始され、かつアクティブでなくなったら完了とみなす
                            playback_finished_event.set()
                            break
                        # 3FPS程度で監視 (0.33秒間隔)
                        await asyncio.sleep(0.33)

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
                        # 再生完了を待つタスク（イベント）
                        wait_playback_task = asyncio.create_task(playback_finished_event.wait())
                        
                        # 待機タスクのリスト
                        tasks_to_wait = [next_packet_task, wait_playback_task]
                        
                        # 応答待ちタイマーがあれば追加
                        if response_timer_task:
                            tasks_to_wait.append(response_timer_task)

                        # 完了を待つ
                        done, pending = await asyncio.wait(
                            tasks_to_wait,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        # 応答待ちタイマーが発火した場合
                        if response_timer_task in done:
                            print(f"\n[Gemini] Response timeout ({config.RESPONSE_TIMEOUT_SECONDS}s). Resetting session.")
                            Logger.log_interaction_result("Response Timeout")
                            self.reset_trigger = True
                            raise SessionResetException()

                        # 再生完了イベントが発火した場合
                        if wait_playback_task in done:
                            print("\n[Gemini] 音声再生が完了したため、受信ループを強制終了します。")
                            # 受信タスクをキャンセル
                            next_packet_task.cancel()
                            try:
                                await next_packet_task
                            except asyncio.CancelledError:
                                pass
                            next_packet_task = None # キャンセル済みとしてマーク
                            break # ループを抜ける

                        # パケット受信が完了した場合
                        if next_packet_task in done:
                            # 待機タスクはキャンセル（次のループで再作成されるため）
                            wait_playback_task.cancel()
                            try:
                                response = next_packet_task.result()
                            except StopAsyncIteration:
                                next_packet_task = None
                                break # ストリーム終了
                            except Exception as e:
                                print(f"Error receiving packet: {e}")
                                next_packet_task = None
                                break
                            
                            next_packet_task = None # 処理完了

                            # --- 以下、既存のレスポンス処理ロジック ---
                            text = None
                            audio_data = getattr(response, "data", None)

                            server_content = getattr(response, "server_content", None)
                            if server_content is not None:
                                input_tx = getattr(server_content, "input_transcription", None)
                                if input_tx:
                                    # 新しいターンが始まったとみなしてフラグをリセット
                                    if has_received_content:
                                        has_received_content = False
                                        is_playing = False
                                    
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
                                    # モデル発話終了
                                    break
                                continue

                            if not has_received_content:
                                has_received_content = True
                                
                                # 応答が来たのでタイマーをキャンセル
                                if response_timer_task:
                                    response_timer_task.cancel()
                                    response_timer_task = None
                                
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
                                    Logger.log_gemini_conversation("User (Audio)", combined_transcript)
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
                                        is_playing = True
                                        await asyncio.to_thread(update_pose, "default")
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
                                # Voicevoxの場合は再生開始時にdefaultに戻すため、ここでは戻さない
                                if not config.USE_VOICEVOX:
                                    await asyncio.to_thread(update_pose, "default")

                            if text:
                                full_response_text.append(text)
                                
                                # Handle [CAPTURE_IMAGE]
                                parts_to_process = []
                                if "[CAPTURE_IMAGE]" in text:
                                    raw_parts = text.split("[CAPTURE_IMAGE]")
                                    for i, part in enumerate(raw_parts):
                                        parts_to_process.append({"text": part, "action": None})
                                        if i < len(raw_parts) - 1:
                                            parts_to_process.append({"text": "", "action": "enable_capture"})
                                else:
                                    parts_to_process.append({"text": text, "action": None})
                                
                                for item in parts_to_process:
                                    sub_text = item["text"]
                                    action = item["action"]
                                    
                                    if action == "enable_capture":
                                        print("\n!!! 撮影トリガーを検出しました (Stream) - Waiting for newline !!!")
                                        await asyncio.to_thread(update_pose, "pic")
                                        capture_pending = True
                                        continue
                                    
                                    clean_sub_text = sub_text.replace("[End_Talk]", "")
                                    
                                    if capture_pending:
                                        if '\n' in clean_sub_text:
                                            print(" [Debug] Capture triggered at newline")
                                            pre, post = clean_sub_text.split('\n', 1)
                                            
                                            if pre:
                                                print(pre, end="", flush=True)
                                                if config.USE_VOICEVOX and player:
                                                    player.add_text(pre)
                                            
                                            print('\n', end="", flush=True)
                                            if config.USE_VOICEVOX and player:
                                                player.add_text('\n')
                                                player.add_event(capture_callback)
                                            else:
                                                await perform_capture()
                                            
                                            capture_pending = False
                                            
                                            if post:
                                                print(post, end="", flush=True)
                                                if config.USE_VOICEVOX and player:
                                                    player.add_text(post)
                                        else:
                                            if clean_sub_text:
                                                print(clean_sub_text, end="", flush=True)
                                                if config.USE_VOICEVOX and player:
                                                    player.add_text(clean_sub_text)
                                    else:
                                        if clean_sub_text:
                                            print(clean_sub_text, end="", flush=True)
                                            if config.USE_VOICEVOX and player:
                                                player.add_text(clean_sub_text)
                                    
                                    if "[End_Talk]" in sub_text:
                                        end_talk_detected = True
                            
                            if not config.USE_VOICEVOX and audio_out_stream and audio_data:
                                if not is_playing:
                                    is_playing = True
                                    await asyncio.to_thread(update_pose, "default")
                                await asyncio.to_thread(audio_out_stream.write, audio_data)
                            
                            if server_content and getattr(server_content, "generation_complete", False): #geminiから生成される音声を使用したフラグ判定を行いたい場合は、turn_completeを使う。
                                break

                    # --- TURN IS COMPLETE ---
                    # 残りのトランスクリプトがあれば出力
                    if user_transcript_buffer:
                        if not print_user_header:
                             print()
                        
                        combined_transcript = "".join(user_transcript_buffer)
                        # print(f"[Gemini] User Transcript: {combined_transcript}")
                        Logger.log_gemini_conversation("User (Audio)", combined_transcript)
                        user_transcript_buffer = []

                        # ユーザー発話終了後、考え中ポーズに移行
                        if not capture_pending:
                            await asyncio.to_thread(update_pose, "thinking")
                        
                        # ユーザー発話が確定した時点で、応答待ちタイマーを開始
                        if not has_received_content:
                            if response_timer_task:
                                response_timer_task.cancel()
                            response_timer_task = asyncio.create_task(asyncio.sleep(config.RESPONSE_TIMEOUT_SECONDS))

                    if has_received_content:
                        final_text = "".join(full_response_text)
                        
                        # ログ記録: Geminiの応答
                        Logger.log_gemini_conversation("Gemini", final_text)
                        
                        if capture_pending:
                            print("\n!!! 撮影トリガーを検出しました (End of Turn - Fallback) !!!")
                            if config.USE_VOICEVOX and player:
                                player.add_event(capture_callback)
                            else:
                                await perform_capture()
                        
                        final_text_for_display = final_text.replace("[CAPTURE_IMAGE]", "").replace("[End_Talk]", "").strip()
                        
                        print("\n\n--- [Full Response Captured] ---")

                        if config.USE_VOICEVOX and player:
                            player.finish()
                            # 既に再生完了イベントで抜けてきた場合は wait_done は即座に終わるはずだが、念のため呼ぶ
                            await asyncio.to_thread(player.wait_done)
                            print("\n[Gemini] 音声再生が完了しました。")
                        
                        # End_Talk の処理を再生完了後に移動
                        if end_talk_detected:
                            print("\n[Gemini] End of conversation detected. Enforcing 15s cooldown.")
                            
                            # ログ記録: 会話終了
                            Logger.log_interaction_result("Conversation Ended (End_Talk)")

                            should_resume_mic = False # マイク再開を抑制
                            if self.yolo_detector:
                                # 15秒間検出をブロックするように last_detection_time を設定
                                # 確実に15秒後に反応できるように、計算上は少し短め(12秒)に設定する
                                self.yolo_detector.last_detection_time = time.time() - config.DETECTION_INTERVAL + 12
                            
                            raise SessionResetException()

                finally:
                    if monitor_task:
                        monitor_task.cancel()
                    
                    if response_timer_task:
                        response_timer_task.cancel()
                    
                    if self.reset_trigger:
                        should_resume_mic = False

                    if should_resume_mic:
                        self.mic_is_active.set()
                        print("--- Mic resumed ---")
                    else:
                        self.mic_is_active.clear()
                        print("--- Mic kept muted (End_Talk/Timeout) ---")

                    await asyncio.to_thread(update_pose, "default")

                    # YOLOを通常FPSモードに戻す
                    if self.yolo_detector:
                        self.yolo_detector.set_low_fps_mode(False)

                    if player:
                        player.close()
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
                audio_out_stream.close()


    async def run(self):
        """メインのタスクグループを立ち上げる"""
        global genai, types, client
        
        t_start_import = time.perf_counter()
        print("[Gemini] Google GenAI ライブラリをインポート中...")
        
        from google import genai
        from google.genai import types
        
        print(f"[Gemini] GenAI インポート完了: {time.perf_counter() - t_start_import:.3f}s")

        client = genai.Client(http_options={"api_version": "v1beta"})

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
                async with (
                    client.aio.live.connect(model=MODEL, config=live_config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.out_queue = asyncio.Queue(maxsize=5)

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

            while True:
                # セッション開始待機
                print("[Gemini] Waiting for person detection to start session...")
                self.session_active.clear()
                await self.session_active.wait()
                
                try:
                    # 接続試行
                    if vad_config:
                        try:
                            print("[Gemini] Connecting with VAD settings...")
                            await start_session(config_with_vad)
                        except Exception as e:
                            if is_reset_exception(e):
                                # セッションリセット時はループの先頭に戻り、再度検出待ちになる
                                print("[Gemini] Session ended. Returning to detection wait state.")
                                self.session_active.clear()
                                self.detection_triggered = False
                                self.mic_is_active.clear()
                                # YOLOの通知フラグをリセット
                                if self.yolo_detector:
                                    self.yolo_detector.reset_notification_flag()
                                continue
                            
                            # ValidationError かどうかを判定 (型名やメッセージで)
                            error_str = str(e)
                            if "ValidationError" in str(type(e).__name__) or "Extra inputs are not permitted" in error_str:
                                print(f"[Gemini] VAD settings not supported by this SDK version. Falling back to default.")
                                # print(f"  -> Error details: {error_str.splitlines()[0]}...") # 最初の一行だけ表示 (抑制)
                                await start_session(config_no_vad)
                            elif isinstance(e, (TimeoutError, OSError)) or "TimeoutError" in type(e).__name__:
                                print(f"[Gemini] Connection error: {e}. Retrying in 5 seconds...")
                                if self.audio_stream:
                                    try:
                                        self.audio_stream.stop_stream()
                                        self.audio_stream.close()
                                    except Exception:
                                        pass
                                    self.audio_stream = None
                                self.session_active.clear()
                                self.detection_triggered = False
                                self.mic_is_active.clear()
                                # YOLOの通知フラグをリセット
                                if self.yolo_detector:
                                    self.yolo_detector.reset_notification_flag()
                                await asyncio.sleep(5)
                                continue
                            else:
                                raise e
                    else:
                        try:
                            await start_session(config_no_vad)
                        except (TimeoutError, OSError) as e:
                            print(f"[Gemini] Connection error (No VAD): {e}. Retrying in 5 seconds...")
                            if self.audio_stream:
                                try:
                                    self.audio_stream.stop_stream()
                                    self.audio_stream.close()
                                except Exception:
                                    pass
                                self.audio_stream = None
                            self.session_active.clear()
                            self.detection_triggered = False
                            self.mic_is_active.clear()
                            # YOLOの通知フラグをリセット
                            if self.yolo_detector:
                                self.yolo_detector.reset_notification_flag()
                            await asyncio.sleep(5)
                            continue
                    
                    break # Normal exit

                except SessionResetException:
                    print("[Gemini] Session ended. Returning to detection wait state.")
                    if self.audio_stream:
                        try:
                            self.audio_stream.stop_stream()
                            self.audio_stream.close()
                        except Exception:
                            pass
                        self.audio_stream = None
                    self.session_active.clear()
                    self.detection_triggered = False
                    self.mic_is_active.clear()
                    # YOLOの通知フラグをリセット
                    if self.yolo_detector:
                        self.yolo_detector.reset_notification_flag()
                    await asyncio.sleep(1)
                    continue
                except Exception as e:
                    if is_reset_exception(e):
                        print("[Gemini] Session ended. Returning to detection wait state.")
                        if self.audio_stream:
                            try:
                                self.audio_stream.stop_stream()
                                self.audio_stream.close()
                            except Exception:
                                pass
                            self.audio_stream = None
                        self.session_active.clear()
                        self.detection_triggered = False
                        self.mic_is_active.clear()
                        # YOLOの通知フラグをリセット
                        if self.yolo_detector:
                            self.yolo_detector.reset_notification_flag()
                        await asyncio.sleep(1)
                        continue
                    
                    if isinstance(e, asyncio.CancelledError):
                        break
                    
                    if isinstance(e, (TimeoutError, OSError)) or "TimeoutError" in type(e).__name__:
                        print(f"[Gemini] Connection error (Outer): {e}. Retrying in 5 seconds...")
                        if self.audio_stream:
                            try:
                                self.audio_stream.stop_stream()
                                self.audio_stream.close()
                            except Exception:
                                pass
                            self.audio_stream = None
                        self.session_active.clear()
                        self.detection_triggered = False
                        self.mic_is_active.clear()
                        # YOLOの通知フラグをリセット
                        if self.yolo_detector:
                            self.yolo_detector.reset_notification_flag()
                        await asyncio.sleep(5)
                        continue
                    
                    print(f"[Gemini] Unexpected error: {e}")
                    traceback.print_exc()
                    print("[Gemini] Resetting session and retrying in 5 seconds...")
                    
                    if self.audio_stream:
                        try:
                            self.audio_stream.stop_stream()
                            self.audio_stream.close()
                        except Exception:
                            pass
                        self.audio_stream = None

                    self.session_active.clear()
                    self.detection_triggered = False
                    self.mic_is_active.clear()
                    # YOLOの通知フラグをリセット
                    if self.yolo_detector:
                        self.yolo_detector.reset_notification_flag()
                    await asyncio.sleep(5)
                    continue

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
