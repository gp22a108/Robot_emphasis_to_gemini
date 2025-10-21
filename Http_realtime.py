# Http_realtime.py (修正後)

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import json # jsonモジュールをインポート
from socketserver import ThreadingMixIn

# --- グローバル変数 ---
pose_data = {
    "CSotaMotion.SV_R_SHOULDER": 0,
    "CSotaMotion.SV_R_ELBOW": 0,
    "CSotaMotion.SV_L_SHOULDER": 0,
    "CSotaMotion.SV_L_ELBOW": 0,
    "CSotaMotion.SV_HEAD_Y": 0,
    "CSotaMotion.SV_HEAD_R": 0,
    "CSotaMotion.SV_BODY_Y": 0,
    "CSotaMotion.SV_HEAD_P": 0,
}
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
        if self.path == '/pose':
            content_length = int(self.headers['Content-Length'])
            post_body = self.rfile.read(content_length)
            
            try:
                received_data = json.loads(post_body)
                
                with data_lock:
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
