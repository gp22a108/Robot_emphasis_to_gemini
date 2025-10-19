import io
import re
import sys
import wave
from concurrent.futures import ThreadPoolExecutor

import pyaudio
import requests

# VOICEVOX EngineのベースURL
BASE_URL = "http://localhost:50021"
# リクエストのタイムアウト時間（秒）
REQUEST_TIMEOUT = 10
SYNTHESIS_TIMEOUT = 60  # 音声合成は時間がかかる場合があるので長めに設定

# オーディオフォーマットを話者IDごとにキャッシュするための辞書
_audio_format_cache = {}

def generate_audio(session: requests.Session, text: str, speaker: int) -> bytes:
    """
    VOICEVOXエンジンにリクエストを送り、音声データ（WAV形式）を生成する。

    Args:
        text (str): 合成するテキスト。
        speaker (int): 話者ID。

    Returns:
        bytes: WAV形式の音声データ。

    Raises:
        requests.exceptions.RequestException: HTTPリクエストに失敗した場合。
    """
    # 1. audio_query: テキストから音声合成用のクエリを作成
    query_params = {"text": text, "speaker": speaker}
    query_response = session.post(
        f"{BASE_URL}/audio_query",
        params=query_params,
        timeout=REQUEST_TIMEOUT
    )
    query_response.raise_for_status()  # エラーがあれば例外を発生させる
    audio_query = query_response.json()

    # 2. synthesis: クエリを使って音声合成を実行
    synth_params = {"speaker": speaker}
    synth_response = session.post(
        f"{BASE_URL}/synthesis",
        params=synth_params,
        json=audio_query,
        timeout=SYNTHESIS_TIMEOUT
    )
    synth_response.raise_for_status()

    return synth_response.content

def get_audio_format(session: requests.Session, speaker: int) -> tuple[int, int, int]:
    """
    オーディオフォーマット（サンプルレート、チャンネル数など）を取得します。
    初回はダミーの音声データを生成して取得し、結果をキャッシュします。

    Args:
        speaker (int): 話者ID。

    Returns:
        tuple: (チャンネル数, サンプル幅, フレームレート)
    """
    # まずキャッシュを確認
    if speaker in _audio_format_cache:
        return _audio_format_cache[speaker]

    print("オーディオフォーマットを新規取得中...")
    # 短いテキストで一度音声を生成し、WAVヘッダーからフォーマット情報を読み取る
    dummy_wav_data = generate_audio(session, "あ", speaker)
    with wave.open(io.BytesIO(dummy_wav_data), 'rb') as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        format_info = (channels, sampwidth, framerate)
        _audio_format_cache[speaker] = format_info # 結果をキャッシュに保存
        return format_info

def play_text(text: str, speaker: int = 1):
    """
    テキストをチャンクに分割し、音声を先読みしながらリアルタイムで再生します。

    Args:
        text (str): 再生するテキスト。
        speaker (int, optional): 話者ID. Defaults to 1.
    """

    # 文章を句読点や改行で分割 (Javaの正規表現をPythonで再現)
    chunks = [s.strip() for s in re.split(r'(?<=[。！？\n])|(?=\n)', text) if s.strip()]

    if not chunks:
        print("テキストが空です。")
        return

    p = None
    stream = None
    try:
        # セッションを開始し、接続を再利用する
        with requests.Session() as session:
            # 最初にオーディオフォーマットを取得
            channels, sampwidth, framerate = get_audio_format(session, speaker)
            print(f"フォーマット: {framerate}Hz, {sampwidth*8}bit, {channels}ch")

            # PyAudioの初期化とストリームの準備
            p = pyaudio.PyAudio()
            # 再生を開始せずにストリームを準備し、最初のデータ書き込み直前に開始する
            stream = p.open(format=p.get_format_from_width(sampwidth),
                            channels=channels,
                            rate=framerate,
                            output=True,
                            start=False)

            # シングルスレッドのExecutorで音声生成を非同期に実行
            with ThreadPoolExecutor(max_workers=1) as executor:
                # 最初のチャンクの音声生成を非同期で開始
                next_audio_future = executor.submit(generate_audio, session, chunks[0], speaker)
                stream_started = False

                for i, chunk in enumerate(chunks):
                    # 現在のチャンクの音声データが生成されるのを待つ
                    current_wav_data = next_audio_future.result()

                    # 最初のデータを書き込む直前にストリームを開始
                    if not stream_started:
                        stream.start_stream()
                        stream_started = True

                    # 次のチャンクがあれば、非同期で音声生成を開始（先読み）
                    if i + 1 < len(chunks):
                        next_audio_future = executor.submit(generate_audio, session, chunks[i + 1], speaker)

                    # 現在のチャンクを再生
                    sys.stdout.write(f"\r再生中: {chunk[:40]}...")
                    sys.stdout.flush()
                    with wave.open(io.BytesIO(current_wav_data), 'rb') as wf:
                        data = wf.readframes(1024)
                        while data:
                            stream.write(data)
                            data = wf.readframes(1024)

        print("\n再生が完了しました。")

    except requests.exceptions.RequestException as e:
        print(f"\nVOICEVOXエンジンへの接続に失敗しました: {e}")
        print("VOICEVOXエンジンが起動しているか、URLが正しいか確認してください。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
    finally:
        # エラー発生時も必ずリソースを解放する
        if stream:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
        print("\n処理を終了します。")

def main():
    """
    このスクリプトを直接実行した際のメイン処理。
    コマンドライン引数からテキストを受け取るか、デフォルトのテキストを再生します。
    """
    # コマンドライン引数があればそれをテキストとして使い、なければデフォルトのテキストを使う
    if len(sys.argv) > 1:
        text_to_play = " ".join(sys.argv[1:])
    else:
        text_to_play = "Hello, World!"

    speaker_id = 1  # 話者IDは適宜変更してください
    try:
        play_text(text_to_play, speaker=speaker_id)
    except KeyboardInterrupt:
        # Ctrl+Cで中断された場合、メッセージを出力して終了
        print("\n再生を中断しました。")

if __name__ == "__main__":
    main()