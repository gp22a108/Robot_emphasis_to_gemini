import sys
import re
import queue
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, wait
import traceback
from typing import Callable, Optional
import config  # 設定ファイルをインポート

# --- 定数 ---
BASE_URL = config.BASE_URL
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.CHANNELS
WAV_HEADER_SIZE = config.WAV_HEADER_SIZE
MAX_WORKERS = 6  # 同時処理数の設定


def generate_audio_chunk(session: requests.Session, text: str, speaker: int, index: int) -> tuple[int, bytes]:
    """VOICEVOXエンジンで音声を生成し、(インデックス, バイナリデータ)を返す"""
    if not text:
        return (index, b"")

    synthesis_timeout = 15 + len(text) // 5

    try:
        query_params = {"text": text, "speaker": speaker}
        # audio_query のタイムアウトを少し長めに確保
        query_res = session.post(f"{BASE_URL}/audio_query", params=query_params, timeout=max(REQUEST_TIMEOUT, 5))
        query_res.raise_for_status()

        audio_query = query_res.json()
        audio_query["speedScale"] = config.SPEED_SCALE

        synth_res = session.post(
            f"{BASE_URL}/synthesis",
            params={"speaker": speaker},
            json=audio_query,
            timeout=synthesis_timeout
        )
        synth_res.raise_for_status()
        return (index, synth_res.content)

    except Exception as e:
        print(f"\n[Error] 生成エラー ({text[:10]}...): {e}")
        return (index, b"")


def _create_chunks(text: str) -> list[str]:
    """テキストを句読点で分割する（テスト用・一括変換用）"""
    text = text.strip()
    if not text:
        return []

    # 句読点（、。！？）または改行で分割する。ただし、分割記号はチャンクの末尾に含める。
    split_pattern = r'(?<=[、。！？\n])'
    parts = re.split(split_pattern, text)

    chunks = []
    for part in parts:
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks


class VoicevoxStreamPlayer:
    def __init__(self, speaker: int, on_last_chunk_start: Optional[Callable[[], None]] = None, on_play_start: Optional[Callable[[], None]] = None):
        self.speaker = speaker
        self.on_last_chunk_start = on_last_chunk_start
        self.on_play_start = on_play_start
        self.is_connected = False
        self._text_buffer = ""  # ストリーミング用テキストバッファ

        self._session = requests.Session()
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._audio_queue = queue.Queue()
        self._text_queue = queue.Queue()
        self._results_buffer = {}
        self._next_index_to_play = 0
        self._next_index_to_generate = 0
        self._is_generation_finished = False
        self._total_chunks_added = 0
        self._last_chunk_callback_triggered = threading.Event()
        self._play_start_triggered = False
        self._is_writing = False # 再生デバイス書き込み中フラグ

        self._playback_ready = threading.Event()
        self._player_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._generation_thread = threading.Thread(target=self._generation_worker, daemon=True)

        # 接続確認を非同期で行うか、タイムアウトを短くする
        if not self._is_voicevox_running():
            self._executor.shutdown()
            return

        self._player_thread.start()
        self._generation_thread.start()
        self._playback_ready.wait(timeout=2.0)
        
        # 再生スレッドが正常に開始したか確認
        if not self._playback_ready.is_set() or not self._player_thread.is_alive():
            self.close()
            print("[Voicevox Error] Failed to initialize audio playback stream.")
            return
        
        self.is_connected = True

    @property
    def has_started_playing(self) -> bool:
        """再生が開始されたかどうかを返す"""
        return self._play_start_triggered

    @property
    def is_active(self) -> bool:
        """
        プレーヤーがアクティブかどうかを返す。
        以下のいずれかの場合に True を返す:
        1. テキストキューに未処理のテキストがある
        2. 音声キューに未再生の音声データがある
        3. テキストバッファに未送信のテキストがある
        4. 生成済みだが再生キューに移動していないデータがある (_next_index_to_play < _next_index_to_generate)
        5. 現在音声デバイスに書き込み中である
        """
        return (not self._text_queue.empty()) or \
               (not self._audio_queue.empty()) or \
               (bool(self._text_buffer.strip())) or \
               (self._next_index_to_play < self._next_index_to_generate) or \
               self._is_writing

    def _is_voicevox_running(self) -> bool:
        """Voicevoxエンジンが起動しているか確認する"""
        try:
            response = self._session.get(f"{BASE_URL}/version", timeout=0.5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            print(f"\n[Voicevox Error] Voicevoxエンジンに接続できません。URL: {BASE_URL}")
            return False

    def _playback_worker(self):
        """音声データをキューから受け取り再生するワーカー"""
        import sounddevice as sd

        stream = None
        try:
            stream = sd.RawOutputStream(
                samplerate=SAMPLE_RATE, blocksize=0, channels=CHANNELS,
                dtype='int16', latency='low'
            )
            stream.start()
            self._playback_ready.set()

            while True:
                item = self._audio_queue.get()
                if item is None:
                    # 終了時にまだトリガーされていなければ実行 (Fallback)
                    if self.on_last_chunk_start and not self._last_chunk_callback_triggered.is_set():
                         # print("\n--- 全再生終了、コールバックをトリガー(Fallback) ---")
                         self.on_last_chunk_start()
                         self._last_chunk_callback_triggered.set()

                    self._audio_queue.task_done()
                    break
                
                try:
                    index, pcm_data = item
                    
                    if not self._play_start_triggered and pcm_data:
                        self._play_start_triggered = True
                        if self.on_play_start:
                            # print("\n--- 再生開始、コールバックをトリガー ---")
                            self.on_play_start()

                    # 再生前チェック
                    if self.on_last_chunk_start and self._is_generation_finished and index == self._total_chunks_added - 1:
                        if not self._last_chunk_callback_triggered.is_set():
                            # print("\n--- 最後のチャンクの再生を開始、コールバックをトリガー ---")
                            self.on_last_chunk_start()
                            self._last_chunk_callback_triggered.set()

                    if pcm_data:
                        self._is_writing = True
                        stream.write(pcm_data)
                        self._is_writing = False

                    # 再生後チェック
                    if self.on_last_chunk_start and self._is_generation_finished and index == self._total_chunks_added - 1:
                        if not self._last_chunk_callback_triggered.is_set():
                            # print("\n--- 最後のチャンクの再生完了、コールバックをトリガー(Post-play) ---")
                            self.on_last_chunk_start()
                            self._last_chunk_callback_triggered.set()

                except Exception as e:
                    print(f"\n[Voicevox Error] 再生中のエラー: {e}")
                    self._is_writing = False
                finally:
                    self._audio_queue.task_done()

        except Exception as e:
            print(f"\n[Voicevox Error] 再生エラー: {e}")
            if not self._playback_ready.is_set():
                 self._playback_ready.set()
        finally:
            if stream:
                stream.stop()
                stream.close()

    def _flush_results_buffer(self):
        """バッファ内の完了したチャンクを順番に再生キューへ移動する"""
        while self._next_index_to_play in self._results_buffer:
            pcm_data = self._results_buffer.pop(self._next_index_to_play)
            # コンソール出力を抑制して高速化
            # total_chunks_str = f"{self._total_chunks_added}" if self._is_generation_finished else "?"
            # try:
            #     sys.stdout.write(f"\r再生中 ({self._next_index_to_play + 1}/{total_chunks_str})...")
            #     sys.stdout.flush()
            # except Exception:
            #     pass
            self._audio_queue.put((self._next_index_to_play, pcm_data))
            self._next_index_to_play += 1

    def _generation_worker(self):
        """テキストチャンクを処理し、音声生成タスクを管理するワーカー"""
        future_to_index = {}
        while True:
            try:
                try:
                    text_chunk = self._text_queue.get(timeout=0.05)
                    if text_chunk is None:
                        self._is_generation_finished = True
                        self._total_chunks_added = self._next_index_to_generate
                        self._text_queue.task_done()
                        break 
                    
                    try:
                        index = self._next_index_to_generate
                        future = self._executor.submit(generate_audio_chunk, self._session, text_chunk, self.speaker, index)
                        future_to_index[future] = index
                        self._next_index_to_generate += 1
                    finally:
                        self._text_queue.task_done()
                except queue.Empty:
                    pass

                if future_to_index:
                    done, not_done = wait(future_to_index.keys(), timeout=0.01)
                    for future in done:
                        index = future_to_index.pop(future)
                        try:
                            _, wav_data = future.result()
                            if wav_data and len(wav_data) > WAV_HEADER_SIZE:
                                self._results_buffer[index] = wav_data[WAV_HEADER_SIZE:]
                            else:
                                self._results_buffer[index] = None
                        except Exception as e:
                            print(f"[Error] Future result error (index {index}): {e}")
                            self._results_buffer[index] = None
                    
                    self._flush_results_buffer()

            except Exception as e:
                print(f"[Fatal Error] Generation worker crashed: {e}")
                traceback.print_exc()
                break

        if future_to_index:
            done, not_done = wait(future_to_index.keys())
            for future in done:
                index = future_to_index.pop(future)
                try:
                    _, wav_data = future.result()
                    if wav_data and len(wav_data) > WAV_HEADER_SIZE:
                        self._results_buffer[index] = wav_data[WAV_HEADER_SIZE:]
                    else:
                        self._results_buffer[index] = None
                except Exception:
                    self._results_buffer[index] = None
            self._flush_results_buffer()

    def add_text(self, text: str):
        """再生するテキストチャンクを追加する（句読点待ちバッファリング付き）"""
        if not self.is_connected: return
        if self._is_generation_finished:
            print("Warning: Player is already finished. Cannot add more text.")
            return
        
        self._text_buffer += text
        
        # 句読点（、。！？）または改行で分割
        split_pattern = r'(?<=[、。！？\n])'
        parts = re.split(split_pattern, self._text_buffer)
        
        for part in parts[:-1]:
            if part.strip():
                self._text_queue.put(part.strip())
        
        self._text_buffer = parts[-1]

    def finish(self):
        """テキストの追加が完了したことを通知する"""
        if not self.is_connected: return
        
        if self._text_buffer.strip():
            self._text_queue.put(self._text_buffer.strip())
        self._text_buffer = ""
            
        self._text_queue.put(None)

    def wait_done(self):
        """すべての音声の生成と再生が完了するのを待つ"""
        if not self.is_connected: return

        # 全てのテキストがキューに追加され、生成スレッドで処理されるのを待つ
        self._text_queue.join()
        if self._generation_thread and self._generation_thread.is_alive():
            self._generation_thread.join()

        # 全ての音声データが再生キューに追加され、再生スレッドに渡されるのを待つ
        self._audio_queue.join()

    def close(self):
        """リソースを解放する"""
        if hasattr(self, '_text_queue') and self._text_queue:
            self._text_queue.put(None)
        if hasattr(self, '_audio_queue') and self._audio_queue:
            self._audio_queue.put(None)
        
        if hasattr(self, '_player_thread') and self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=1)
        
        if hasattr(self, '_executor') and self._executor:
             self._executor.shutdown(wait=False, cancel_futures=True)
        
        if hasattr(self, '_session') and self._session:
            self._session.close()


def play_text(text: str, speaker: int = 1, on_last_chunk_start: Optional[Callable[[], None]] = None):
    """[互換性のため] テキスト全体を受け取って再生する"""
    player = None
    try:
        player = VoicevoxStreamPlayer(speaker=speaker, on_last_chunk_start=on_last_chunk_start)
        if player.is_connected:
            chunks = _create_chunks(text)
            for chunk in chunks:
                player._text_queue.put(chunk)
            player.finish()
            player.wait_done()
            player.close()
    except Exception as e:
        print(f"An unexpected error occurred in play_text: {e}")
        if player:
            player.close()


def play_text_async(text: str, speaker: int = 1, on_last_chunk_start: Optional[Callable[[], None]] = None):
    """
    play_textを別スレッドで実行し、メインスレッド（YOLO等）をブロックせずに音声を再生する。
    """
    threading.Thread(
        target=play_text,
        args=(text, speaker, on_last_chunk_start),
        daemon=True
    ).start()


def main():
    """テスト用のメイン関数"""
    try:
        text_to_play = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
            "これはストリーミング再生のテストです。テキストが、このように、少しずつ、追加されても、スムーズに、再生されるはずです。"

        def example_callback():
            print("\n*** ラストチャンクの再生が始まりました！ ***")

        print("--- [Test 1] play_text (互換モード) ---")
        play_text(text_to_play, speaker=config.SPEAKER_ID, on_last_chunk_start=example_callback)
        
        print("\n\n--- [Test 2] VoicevoxStreamPlayer (ストリーミングモード) ---")
        
        player = VoicevoxStreamPlayer(speaker=config.SPEAKER_ID, on_last_chunk_start=example_callback)
        if not player.is_connected:
            print("プレーヤーが接続されていないため、テストをスキップします。")
            return

        parts = ["これは", "ストリーミング", "再生の", "テストです。", "テキストが、", "このように、", "少しずつ、", "追加されても、", "スムーズに、", "再生されるはずです。"]
        
        for part in parts:
            if part.strip():
                print(f"  -> Adding chunk: '{part.strip()}'")
                player.add_text(part)
                time.sleep(0.5)
        
        player.finish()
        player.wait_done()
        player.close()

    except KeyboardInterrupt:
        print("\n再生を中断しました。")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
