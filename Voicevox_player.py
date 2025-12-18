import sys
import re
import queue
import threading
import time
import requests
import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
import config  # 設定ファイルをインポート

# --- 定数 ---
BASE_URL = config.BASE_URL
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.CHANNELS
WAV_HEADER_SIZE = config.WAV_HEADER_SIZE


def generate_audio_chunk(session: requests.Session, text: str, speaker: int, index: int) -> tuple[int, bytes]:
    """VOICEVOXエンジンで音声を生成し、(インデックス, バイナリデータ)を返す"""
    if not text:
        return (index, b"")

    synthesis_timeout = 15 + len(text) // 5

    try:
        query_params = {"text": text, "speaker": speaker}
        query_res = session.post(f"{BASE_URL}/audio_query", params=query_params, timeout=REQUEST_TIMEOUT)
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
    """テキストを句読点で分割する"""
    text = text.strip()
    if not text:
        return []

    split_pattern = r'(?<=[、。！？\n])'
    parts = re.split(split_pattern, text)

    chunks = []
    for part in parts:
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks


def playback_worker(audio_queue: queue.Queue, ready_event: threading.Event):
    """
    再生ワーカー: ストリームを先に開いてから、データの到着を待つ
    """
    stream = None

    try:
        stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=0,
            channels=CHANNELS,
            dtype='int16',
            latency='high'
        )
        stream.start()
        time.sleep(0.1)
        ready_event.set()

        while True:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:
                audio_queue.task_done()
                break

            if len(chunk) > WAV_HEADER_SIZE:
                pcm_data = chunk[WAV_HEADER_SIZE:]
                stream.write(pcm_data)

            audio_queue.task_done()

    except Exception as e:
        print(f"\n再生エラー: {e}")
        ready_event.set()
    finally:
        if stream:
            time.sleep(0.2)
            stream.stop()
            stream.close()


def is_voicevox_running(session: requests.Session) -> bool:
    """Voicevoxエンジンが起動しているか確認する"""
    try:
        response = session.get(f"{BASE_URL}/version", timeout=1)
        response.raise_for_status()
        print(f"[Voicevox] エンジン接続成功 (Version: {response.text})")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n[Voicevox Error] Voicevoxエンジンに接続できません。")
        print(f"  - URL: {BASE_URL}")
        print(f"  - エラー: {e}")
        print(f"  - Voicevoxが起動しているか、ポート設定が正しいか確認してください。")
        return False


def play_text(text: str, speaker: int = 1, on_last_chunk_start: Optional[Callable[[], None]] = None):
    """音声生成と再生のメインコントローラ"""
    with requests.Session() as session:
        if not is_voicevox_running(session):
            return  # エンジンがなければ処理を中断

        chunks = _create_chunks(text)
        if not chunks:
            return

        num_chunks = len(chunks)
        audio_queue = queue.Queue()
        ready_event = threading.Event()

        player_thread = threading.Thread(target=playback_worker, args=(audio_queue, ready_event), daemon=True)
        player_thread.start()
        ready_event.wait(timeout=5.0)

        results_buffer = {}
        next_index_to_send = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(generate_audio_chunk, session, chunk, speaker, i)
                future_to_index[future] = i

            for future in as_completed(future_to_index):
                index, wav_data = future.result()

                if wav_data:
                    results_buffer[index] = wav_data
                else:
                    results_buffer[index] = None

                while next_index_to_send in results_buffer:
                    data = results_buffer.pop(next_index_to_send)

                    if data:
                        if next_index_to_send == num_chunks - 1 and on_last_chunk_start:
                            print("\n--- 最後のチャンクの再生を開始、コールバックをトリガー ---")
                            on_last_chunk_start()

                        sys.stdout.write(f"\r再生中 ({next_index_to_send + 1}/{num_chunks})")
                        sys.stdout.flush()
                        audio_queue.put(data)

                    next_index_to_send += 1

    audio_queue.put(None)
    player_thread.join()
    print("\n再生終了")


def main():
    try:
        text_to_play = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
            "これはsounddeviceを使った修正版のコードです。最初の無音が解消され、スムーズに再生されます。"

        def example_callback():
            print("\n*** ラストチャンクの再生が始まりました！ ***")

        play_text(text_to_play, speaker=config.SPEAKER_ID, on_last_chunk_start=example_callback)
    except KeyboardInterrupt:
        print("\n再生を中断しました。")


if __name__ == "__main__":
    main()