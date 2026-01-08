# Http_realtime.py (修正後)

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import json # jsonモジュールをインポート
from socketserver import ThreadingMixIn
import config # 設定ファイルをインポート

# --- グローバル変数 ---
pose_data_default = config.POSE_DATA_DEFAULT.copy()
pose_data_thinking = config.POSE_DATA_THINKING.copy()
pose_data_pic = config.POSE_DATA_PIC.copy()

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

                        try:
                            self.wfile.write(sse_message.encode("utf-8"))
                            self.wfile.flush()
                            last_sent_data_str = current_data_str
                            last_heartbeat_time = time.time()
                        except (ConnectionAbortedError, BrokenPipeError):
                            print("Client disconnected during write.")
                            return

                    # 一定時間データ送信がない場合、ハートビートを送信 (3秒間隔)
                    elif time.time() - last_heartbeat_time > 3.0:
                        try:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                            last_heartbeat_time = time.time()
                        except (ConnectionAbortedError, BrokenPipeError):
                            print("Client disconnected during heartbeat.")
                            return

                    time.sleep(0.1)

            except Exception as e:
                print(f"SSE loop error: {e}")
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
            try:
                content_length = int(self.headers['Content-Length'])
                post_body = self.rfile.read(content_length)

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
                try:
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                except (ConnectionAbortedError, BrokenPipeError):
                    print("Client disconnected before response could be sent.")

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'status': 'error', 'message': 'Invalid JSON'}
                try:
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                except (ConnectionAbortedError, BrokenPipeError):
                    pass
            except Exception as e:
                print(f"POST error: {e}")
                try:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {'status': 'error', 'message': str(e)}
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                except (ConnectionAbortedError, BrokenPipeError):
                    pass
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

def run_server(host="0.0.0.0", port=config.HTTP_SERVER_PORT):
    """HTTPサーバーを指定されたホストとポートで起動する"""
    # Windows環境での0.0.0.0バインディングの問題を回避するため、
    # 明示的に0.0.0.0を指定するか、空文字列を使用する。
    # ここではconfigの値を使用しつつ、デフォルト引数で柔軟性を持たせる。
    
    # 念のためホストが空文字の場合は全てのインターフェースでリッスン
    if not host:
        host = "0.0.0.0"

    with ThreadingHTTPServer((host, port), PoseRequestHandler) as httpd:
        print(f"Serving on http://{host}:{port}")
        httpd.serve_forever()


# --- メイン処理 ---
if __name__ == "__main__":
    run_server()
