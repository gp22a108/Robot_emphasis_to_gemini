# ==========================================
# 設定パラメータ (Configuration)
# ==========================================

# --- モデル設定 ---
# モデルファイルのパス
MODEL_PATH = "yolov12n_openvino_model/yolov12n.xml"
# 使用するデバイス ("CPU", "GPU", "AUTO")
DEVICE_NAME = "AUTO"
# モデルキャッシュのディレクトリ
CACHE_DIR = "./model_cache"
# モデルの入力サイズ
MODEL_INPUT_SIZE = (640, 640)

# --- 検出パラメータ ---
# 信頼度のしきい値
CONFIDENCE_THRESHOLD = 0.5
# Non-Maximum Suppression (NMS) のしきい値
NMS_THRESHOLD = 0.5
# --- 映像入力設定 ---
# 0: デフォルトカメラ, 1: 別カメラ番号, "path/to/video.mp4": 動画ファイル, "rtsp://...": ストリーム
VIDEO_SOURCE = 0
# 同じ物体を再検出するまでの待機時間 (秒)
# Gemini側でセッション管理しているため、YOLO側での制限はほぼなくす
DETECTION_INTERVAL = 1.0

# --- セッション・タイムアウト設定 ---
# 人物が検出されなくなってからセッションを終了するまでの時間 (秒)
# 30秒に延長（移動中などのロスト対策）
SESSION_TIMEOUT_SECONDS = 5
# End_Talk 検出後のクールダウン時間 (秒)
END_TALK_COOLDOWN_SECONDS = 5
# ユーザー発話後、Geminiからの応答を待つ最大時間 (秒)
RESPONSE_TIMEOUT_SECONDS = 20

# --- クラス定義 ---
CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# --- 音声出力設定 ---
# True: Voicevoxを使用, False: Geminiの音声をそのまま使用
USE_VOICEVOX = True

# --- Gemini設定 ---
# VOICEVOXのキャラクターID
SPEAKER_ID = 10002
# VOICEVOX NEMO 男性3→10002
# VOICEVOX 麒ヶ島宗麟→53

# --- Voicevox Player設定 ---
# 再生速度
SPEED_SCALE = 1.2
# VOICEVOXのベースURL
BASE_URL = "http://127.0.0.1:50121"
# リクエストのタイムアウト（秒）
REQUEST_TIMEOUT = 30
# サンプルレート
SAMPLE_RATE = 24000
# チャンネル数
CHANNELS = 1
# WAVヘッダーサイズ
WAV_HEADER_SIZE = 44
# 同時処理数
MAX_WORKERS = 1

# --- Gemini VAD設定 (感度調整) ---
# 発話終了とみなすまでの無音時間 (ミリ秒)
# 値を小さくすると、発話終了の判定が早くなります（キレが良い）。
# 値を大きくすると、発話終了の判定が遅くなります（間を許容する）。
SPEECH_SILENCE_DURATION_MS = 200

# 旧設定 (無効化)
# VAD_POSITIVE_THRESHOLD = 0.0
# VAD_NEGATIVE_THRESHOLD = 1.0

# --- Http_realtime設定 ---
# HTTPサーバーのホスト
HTTP_SERVER_HOST = "0.0.0.0"
# HTTPサーバーのポート
HTTP_SERVER_PORT = 8000

# --- ポーズデータ設定 ---
POSE_DATA_DEFAULT = {   #デフォルトのポーズ
    "CSotaMotion.SV_R_SHOULDER": 700,
    "CSotaMotion.SV_R_ELBOW": 0,
    "CSotaMotion.SV_L_SHOULDER": -150,
    "CSotaMotion.SV_L_ELBOW": -500,
    "CSotaMotion.SV_HEAD_Y": 0,
    "CSotaMotion.SV_HEAD_R": 0,
    #"CSotaMotion.SV_BODY_Y": 0,    sota側のプログラムにswitch関数を使用しているため、無くても動作します。
    #"CSotaMotion.SV_HEAD_P": 0,    この２行はHttp_realtimeに初期値を設定しています。上書きを避けるためにコメントアウトしてください。sotaプログラム起動時に0で初期化されます。
}

POSE_DATA_THINKING = {   #考え中のポーズ
    "CSotaMotion.SV_R_SHOULDER": 550,
    "CSotaMotion.SV_R_ELBOW": 700,
    "CSotaMotion.SV_L_SHOULDER": -150,
    "CSotaMotion.SV_L_ELBOW": -580,
    "CSotaMotion.SV_HEAD_Y": 0,
    "CSotaMotion.SV_HEAD_R": -300,
    #"CSotaMotion.SV_BODY_Y": 0,
    #"CSotaMotion.SV_HEAD_P": 0, # 少しうつむくなど変化をつける
}

POSE_DATA_PIC = {   #撮影時のポーズ
    "CSotaMotion.SV_R_SHOULDER": 700,
    "CSotaMotion.SV_R_ELBOW": 0,
    "CSotaMotion.SV_L_SHOULDER": 0,
    "CSotaMotion.SV_L_ELBOW": -700,
    "CSotaMotion.SV_HEAD_Y": 0,
    "CSotaMotion.SV_HEAD_R": 550,
    #"CSotaMotion.SV_BODY_Y": 0,
    #"CSotaMotion.SV_HEAD_P": 0,
}
