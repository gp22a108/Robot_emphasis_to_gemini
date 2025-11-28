import io, re, sys, wave, queue, threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
import pyaudio
import requests

# --- 定数 ---
BASE_URL = "http://127.0.0.1:50121"
REQUEST_TIMEOUT = 10
CHANNELS = 1
SAMPWIDTH = 2
FRAMERATE = 24000

def generate_audio(session: requests.Session, text: str, speaker: int) -> bytes:
    """VOICEVOXエンジンで音声を生成する"""
    # テキスト長に応じてタイムアウトを動的に調整し、長い文章のエラーを防ぐ
    synthesis_timeout = 15 + len(text) // 5

    query_params = {"text": text, "speaker": speaker}
    query_response = session.post(f"{BASE_URL}/audio_query", params=query_params, timeout=REQUEST_TIMEOUT)
    query_response.raise_for_status()

    audio_query = query_response.json()
    audio_query['speedScale'] = 1.2 #再生速度

    synth_response = session.post(f"{BASE_URL}/synthesis", params={"speaker": speaker}, json=audio_query, timeout=synthesis_timeout)
    synth_response.raise_for_status()
    return synth_response.content

def player_worker(stream: pyaudio.Stream, audio_queue: queue.Queue):
    """キューから音声データを取り出し再生するスレッド"""
    # ストリームをウォームアップするために短い無音を再生
    stream.write(b'\x00' * 1024 * SAMPWIDTH * CHANNELS)
    while True:
        wav_data = audio_queue.get()
        if wav_data is None:
            audio_queue.task_done()
            break
        with wave.open(io.BytesIO(wav_data)) as wf:
            for data in iter(lambda: wf.readframes(1024), b''):
                stream.write(data)
        audio_queue.task_done()

def _create_chunks(text: str) -> List[str]:
    """テキストを再生に適したチャンクに分割する"""
    text = text.strip()
    if not text: return []

    match = re.search(r'[、。！？\n]', text)
    if not match: return [text]

    first_end = match.end()
    chunks = [text[:first_end]]
    remaining = text[first_end:].strip()
    if remaining:
        chunks.extend(s.strip() for s in re.split(r'(?<=[。！？\n])', remaining) if s.strip())
    return chunks

def play_text(text: str, speaker: int =10002):
    """音声の生成と再生を並列処理する"""
    chunks = _create_chunks(text)
    if not chunks:
        print("テキストが空です。")
        return

    p, stream = None, None
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(SAMPWIDTH),
                        channels=CHANNELS,
                        rate=FRAMERATE,
                        output=True,
                        frames_per_buffer=1024)

        audio_queue = queue.Queue()
        player_thread = threading.Thread(target=player_worker, args=(stream, audio_queue), daemon=True)
        player_thread.start()

        with requests.Session() as session, ThreadPoolExecutor(max_workers = 2) as executor:
            futures = [executor.submit(generate_audio, session, chunk, speaker) for chunk in chunks]
            for i, future in enumerate(futures):
                wav_data = future.result()
                sys.stdout.write(f"\r再生中 ({i + 1}/{len(chunks)}): {chunks[i][:40]}...")
                sys.stdout.flush()
                audio_queue.put(wav_data)

        audio_queue.put(None)
        audio_queue.join()

    except requests.exceptions.RequestException as e:
        print(f"\nVOICEVOXエンジンへの接続に失敗しました: {e}")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
        print("処理を終了します。")

def main():
    """スクリプトのエントリーポイント"""
    try:
        text_to_play = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
                       "コードを極限まで削減し、可読性を維持した最終バージョンです。"
        play_text(text_to_play, speaker=1)
    except KeyboardInterrupt:
        print("\n再生を中断しました。")

if __name__ == "__main__":
    main()
