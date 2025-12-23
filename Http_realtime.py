# Http_realtime.py (修正後)

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import json # jsonモジュールをインポート
from socketserver import ThreadingMixIn

# --- グローバル変数 ---
pose_data_default = {   #デフォルトのポーズ
    "CSotaMotion.SV_R_SHOULDER": 700,
    "CSotaMotion.SV_R_ELBOW": 0,
    "CSotaMotion.SV_L_SHOULDER": -150,
    "CSotaMotion.SV_L_ELBOW": -500,
    "CSotaMotion.SV_HEAD_Y": 0,
    "CSotaMotion.SV_HEAD_R": 0,
    "CSotaMotion.SV_BODY_Y": 0,
    "CSotaMotion.SV_HEAD_P": 0,
}

pose_data_thinking = {   #考え中のポーズ
    "CSotaMotion.SV_R_SHOULDER": 550,
    "CSotaMotion.SV_R_ELBOW": 700,
    "CSotaMotion.SV_L_SHOULDER": -150,
    "CSotaMotion.SV_L_ELBOW": -580,
    "CSotaMotion.SV_HEAD_Y": 0,
    "CSotaMotion.SV_HEAD_R": -300,
    "CSotaMotion.SV_BODY_Y": 0,
    "CSotaMotion.SV_HEAD_P": 0, # 少しうつむくなど変化をつける
}

pose_data_pic = {   #撮影時のポーズ
    "CSotaMotion.SV_R_SHOULDER": 700,
    "CSotaMotion.SV_R_ELBOW": 0,
    "CSotaMotion.SV_L_SHOULDER": 0,
    "CSotaMotion.SV_L_ELBOW": -700,
    "CSotaMotion.SV_HEAD_Y": 0,
    "CSotaMotion.SV_HEAD_R": 550,
    "CSotaMotion.SV_BODY_Y": 0,
    "CSotaMotion.SV_HEAD_P": 0,
}

# 現在のポーズデータを初期化
pose_data = pose_data_default.copy()

data_lock = threading.Lock()


# --- HTTPサーバー部分 (SSE対応) ---
class PoseRequestHandler(BaseHTTPRequestHandler):
    """
    HTTPリクエストを処理するハンドラ。
    /pose へのGETリクエストに対し、SSEでポーズデータをストリーミング配信する。
    /pose へのPOSTリクエストで、ポーズデータを受け取り更新する。
    """
    def do_GET(self):
        if self.path == "/pose":
            # SSE用のヘッダーを送信
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            print("Client connected for SSE.")
            last_sent_data_str = ""
            last_heartbeat_time = time.time()
            try:
                # 接続が続く限りループ
                while True:
                    with data_lock:
                        ids = list(pose_data.keys())
                        positions = list(pose_data.values())

                    line1 = ",".join(ids)
                    line2 = ",".join(map(str, positions))
                    current_data_str = f"{line1}\n{line2}"

                    # データに変更があった場合のみ送信
                    if current_data_str != last_sent_data_str:
                        # SSEフォーマットでメッセージを構築
                        sse_message = f"data: {line1}\ndata: {line2}\n\n"

                        self.wfile.write(sse_message.encode("utf-8"))
                        self.wfile.flush()
                        last_sent_data_str = current_data_str
                        last_heartbeat_time = time.time()

                    # 一定時間データ送信がない場合、ハートビートを送信 (3秒間隔)
                    elif time.time() - last_heartbeat_time > 3.0:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        last_heartbeat_time = time.time()

                    time.sleep(0.1)

            except BrokenPipeError:
                print("Client disconnected.")
            finally:
                return
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Not Found\n")

    def do_POST(self):
        global pose_data
        if self.path == '/pose':
            content_length = int(self.headers['Content-Length'])
            post_body = self.rfile.read(content_length)

            try:
                received_data = json.loads(post_body)

                with data_lock:
                    # モード指定がある場合
                    if "mode" in received_data:
                        mode = received_data["mode"]
                        if mode == "default":
                            pose_data = pose_data_default.copy()
                            print("Pose switched to DEFAULT")
                        elif mode == "thinking":
                            pose_data = pose_data_thinking.copy()
                            print("Pose switched to THINKING")
                        elif mode == "pic":
                            pose_data = pose_data_pic.copy()
                            print("Pose switched to PIC")
                    else:
                        # 個別の値を更新
                        for key, value in received_data.items():
                            if key in pose_data:
                                pose_data[key] = value

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'success', 'message': 'Pose data updated'}
                self.wfile.write(json.dumps(response).encode('utf-8'))

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'error', 'message': 'Invalid JSON'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'error', 'message': str(e)}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(b'Not Found\n')


    def log_message(self, format, *args):
        # ログ出力を抑制してコンソールをクリーンに保つ
        return

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """マルチスレッド対応のHTTPサーバー"""
    pass

def run_server(host="0.0.0.0", port=8000):
    """HTTPサーバーを指定されたホストとポートで起動する"""
    with ThreadingHTTPServer((host, port), PoseRequestHandler) as httpd:
        print(f"Serving on http://{host}:{port}")
        httpd.serve_forever()


# --- メイン処理 ---
if __name__ == "__main__":
    run_server()
