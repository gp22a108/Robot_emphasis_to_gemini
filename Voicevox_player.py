import sys
import re
import queue
import threading
import requests
import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor

# --- 定数 ---
BASE_URL = "http://127.0.0.1:50121"
REQUEST_TIMEOUT = 10
SAMPLE_RATE = 24000
CHANNELS = 1
# VOICEVOXのWAVヘッダーサイズ（通常44バイト）
WAV_HEADER_SIZE = 44

def generate_audio_chunk(session: requests.Session, text: str, speaker: int) -> bytes:
    """VOICEVOXエンジンで音声を生成し、バイナリデータを返す"""
    if not text:
        return b""

    synthesis_timeout = 15 + len(text) // 5

    try:
        # Audio Queryの作成
        query_params = {"text": text, "speaker": speaker}
        query_res = session.post(f"{BASE_URL}/audio_query", params=query_params, timeout=REQUEST_TIMEOUT)
        query_res.raise_for_status()

        audio_query = query_res.json()
        audio_query["speedScale"] = 1.2 # 再生速度

        # 音声合成（WAV形式で返ってくる）
        synth_res = session.post(
            f"{BASE_URL}/synthesis",
            params={"speaker": speaker},
            json=audio_query,
            timeout=synthesis_timeout
        )
        synth_res.raise_for_status()
        return synth_res.content

    except Exception as e:
        print(f"\n[Error] 生成エラー ({text[:10]}...): {e}")
        return b""

def _create_chunks(text: str) -> list[str]:
    """テキストを句読点で分割する"""
    text = text.strip()
    if not text: return []

    # 句読点で分割し、空文字を除外
    chunks = []
    # 肯定先読みを使って句読点を前の文に含めるなど、より自然な分割が可能ですが
    # ここではシンプルに改行や句読点で分割します
    split_pattern = r'(?<=[、。！？\n])'
    parts = re.split(split_pattern, text)

    for part in parts:
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks

def playback_worker(audio_queue: queue.Queue):
    """
    データが来るまで待機し、データが来たらストリームを開いて再生する
    """
    stream = None

    try:
        # 1. 最初のデータが来るまでここでブロック（待機）する
        # これにより「無音再生」の期間をなくす
        first_chunk = audio_queue.get()

        if first_chunk is None:
            return # データがないまま終了

        # 2. 最初のデータを受け取った瞬間にストリームを開く
        # RawOutputStreamを使用し、ヘッダーを除去したPCMデータを直接流し込む
        stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=1024,
            channels=CHANNELS,
            dtype='int16'
        )
        stream.start()

        # ヘッダー除去処理（WAVの先頭44バイトをスキップ）
        if len(first_chunk) > WAV_HEADER_SIZE:
            stream.write(first_chunk[WAV_HEADER_SIZE:])

        audio_queue.task_done()

        # 3. 2つ目以降のチャンクを順次再生
        while True:
            chunk = audio_queue.get()
            if chunk is None: # 終了シグナル
                audio_queue.task_done()
                break

            if len(chunk) > WAV_HEADER_SIZE:
                stream.write(chunk[WAV_HEADER_SIZE:])

            audio_queue.task_done()

    except Exception as e:
        print(f"\n再生エラー: {e}")
    finally:
        if stream:
            stream.stop()
            stream.close()

def play_text(text: str, speaker: int = 10002):
    """音声生成と再生のメインコントローラ"""
    chunks = _create_chunks(text)
    if not chunks: return

    # 再生用キュー
    audio_queue = queue.Queue()

    # 再生スレッドを開始
    player_thread = threading.Thread(target=playback_worker, args=(audio_queue,), daemon=True)
    player_thread.start()

    # 生成（プロデューサー）処理
    # 最初のチャンクを最優先で取得するため、並列数は調整しても良い
    with requests.Session() as session, ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(generate_audio_chunk, session, chunk, speaker))

        for i, future in enumerate(futures):
            wav_data = future.result()
            if wav_data:
                # 生成できたデータをキューに入れる
                sys.stdout.write(f"\r再生準備完了 ({i + 1}/{len(chunks)})")
                sys.stdout.flush()
                audio_queue.put(wav_data)

    # 終了シグナル
    audio_queue.put(None)

    # 全ての再生が終わるまで待つ
    player_thread.join()
    print("\n再生終了")

def main():
    try:
        text_to_play = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
            "これはsounddeviceを使った修正版のコードです。最初の無音が解消され、スムーズに再生されます。"
        play_text(text_to_play, speaker=1)
    except KeyboardInterrupt:
        print("\n再生を中断しました。")

if __name__ == "__main__":
    main()