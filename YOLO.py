
import time
import threading
import os
import queue
import traceback
import sys
from pathlib import Path
import requests

# 設定ファイルをインポート
import config
# ログ機能をインポート
import Logger

# ==========================================
# 遅延読み込み用のグローバル変数
# ==========================================
cv2 = None
np = None
ov = None
PrePostProcessor = None
ResizeAlgorithm = None
ColorFormat = None
psutil = None
YOLO = None # ultralytics YOLO

def measure_and_print(step_name, start_time, start_cpu, process):
    """処理時間とCPU占有率を計測して表示するヘルパー関数"""
    elapsed_wall = time.perf_counter() - start_time
    elapsed_cpu = time.process_time() - start_cpu
    # interval=Noneで非ブロッキング
    cpu_percent = process.cpu_percent(interval=None)
    print(f"  -> 計測({step_name}): wall={elapsed_wall:.3f}s, cpu={elapsed_cpu:.3f}s, cpu%={cpu_percent:.1f}%")
    return time.perf_counter(), time.process_time()

class YOLOOptimizer:
    def __init__(self, model_path=config.MODEL_PATH, device_name=config.DEVICE_NAME, cache_dir=config.CACHE_DIR, input_size=config.MODEL_INPUT_SIZE, on_detection=None):
        """
        YOLO検出エンジンの初期化
        ここでは重い処理を行わず、run()メソッド内で初期化を行います。
        """
        self.model_path = model_path
        self.device_name = device_name
        self.cache_dir = cache_dir
        self.input_size = input_size
        self.on_detection = on_detection
        
        self.input_width, self.input_height = input_size
        self.last_detection_time = 0
        self.lock = threading.Lock()
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.inference_time_ms = 0.0
        self.thread = None
        self.command_thread = None # ロボット制御用スレッド
        self.watchdog_thread = None # 監視用スレッド
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.current_frame_for_capture = None
        self.last_person_seen_time = 0.0
        self.last_frame_time = 0.0
        self.is_ready_event = threading.Event() # 初期化完了フラグ
        
        # ループ監視用
        self.last_loop_time = time.time()

        # Gemini応答保持用
        self.gemini_response = ""
        self.voicevox_message = ""
        self.gemini_lock = threading.Lock()

        # テキスト描画キャッシュ用
        self.cached_gemini_text = ""
        self.cached_gemini_lines = []

        # 結果受け渡し用キュー
        self.result_queue = queue.Queue(maxsize=2)
        self.face_result_queue = queue.Queue(maxsize=2) # 顔検出結果用キュー
        self.command_queue = queue.Queue(maxsize=1) # ロボット制御コマンド用キュー
        
        # 描画用キャッシュ
        self.latest_results = []
        self.latest_face_results = []
        
        # OpenVINO関連オブジェクト (run内で初期化)
        self.core = None
        self.model = None
        self.compiled_model = None
        self.infer_queue = None
        
        # 顔検出用モデル (YOLOv12n-face)
        self.face_model_size = 'n' # 'n', 's', 'm', 'l'
        # 同じフォルダにある yolov12n-face.pt を使用
        self.face_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'yolov12{self.face_model_size}-face.pt')
        self.face_confidence_threshold = 0.5
        
        # OpenVINO顔検出用
        self.face_net = None
        self.compiled_face_model = None
        self.face_infer_queue = None
        
        # FPS制御用
        self.target_fps = 30.0 # デフォルトFPS
        self.low_fps_mode = False

        # ターゲット追跡用
        self.target_box = None

        # 通知済みフラグ
        self.has_notified_in_session = False
        self.pending_notification_reset = False
        self.last_large_person_time = 0.0
        self._last_camera_timeout_log = 0.0

    def is_ready(self):
        """初期化が完了したかどうかを返す"""
        return self.is_ready_event.is_set()

    def set_low_fps_mode(self, enabled):
        """低FPSモードの切り替え"""
        self.low_fps_mode = enabled
        if enabled:
            self.target_fps = 1.0
            print("[YOLO] Low FPS Mode: ON (1 FPS)")
        else:
            self.target_fps = 30.0
            print("[YOLO] Low FPS Mode: OFF (30 FPS)")

    def reset_notification_flag(self, defer: bool = False):
        """通知済みフラグをリセットする（セッション終了時などに呼ぶ）"""
        try:
            thread_alive = self.thread and self.thread.is_alive()
            Logger.log_system_event("INFO", "YOLO reset_notification_flag ENTRY",
                message=f"defer={defer}, thread_alive={thread_alive}, stop_event={self.stop_event.is_set()}")
            print(f"[YOLO] reset_notification_flag called (defer={defer}, thread_alive={thread_alive})")

            with self.lock:
                if defer:
                    self.pending_notification_reset = True
                    print("[YOLO] Notification flag reset (deferred).")
                    Logger.log_system_event("INFO", "YOLO reset_notification_flag EXIT", message="Deferred mode set")
                    return
                self.has_notified_in_session = False
                self.pending_notification_reset = False
                print("[YOLO] Notification flag reset.")
                Logger.log_system_event("INFO", "YOLO notification flag", message="Flag reset completed")

            Logger.log_system_event("INFO", "YOLO reset_notification_flag EXIT", message="Normal completion")
        except Exception as e:
            Logger.log_system_error("YOLO reset_notification_flag", e)
            print(f"[YOLO Error] Failed to reset notification flag: {e}")
            traceback.print_exc()

    def _initialize_dependencies(self):
        """依存ライブラリとモデルの初期化（別スレッドで実行）"""
        global cv2, np, ov, PrePostProcessor, ResizeAlgorithm, ColorFormat, psutil, YOLO
        
        self.last_loop_time = time.time() # Watchdog reset

        # 既に初期化済みならスキップ
        if self.core is not None:
            return

        print("[YOLO] ライブラリを読み込んでいます...")
        t_start = time.perf_counter()
        
        if cv2 is None:
            import cv2
            import numpy as np
            import openvino as ov
            from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
            import psutil
            from ultralytics import YOLO
        
        print(f"[YOLO] ライブラリ読み込み完了: {time.perf_counter() - t_start:.3f}s")
        self.last_loop_time = time.time() # Watchdog reset

        # --- 計測の準備 ---
        p = psutil.Process()
        p.cpu_percent(None)
        total_start_time = time.perf_counter()
        total_start_cpu = time.process_time()
        last_time, last_cpu = total_start_time, total_start_cpu

        print("[YOLO] OpenVINO Runtimeを起動中...")
        self.core = ov.Core()
        last_time, last_cpu = measure_and_print("OpenVINO Runtime起動", last_time, last_cpu, p)
        self.last_loop_time = time.time() # Watchdog reset

        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.core.set_property({ov.properties.cache_dir: str(cache_path)})
        
        if "GPU" in self.device_name or "AUTO" in self.device_name:
            self.core.set_property(self.device_name, {
                ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.THROUGHPUT
            })

        # --- メインモデルの読み込み ---
        if not os.path.exists(self.model_path):
            message = f"モデルファイルが見つかりません: {os.path.abspath(self.model_path)}"
            Logger.log_system_error("YOLO モデル読み込み", message=message)
            print(f"\n[エラー] {message}")
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"[YOLO] モデルを読み込んでいます: {self.model_path}")
        self.model = self.core.read_model(self.model_path)
        last_time, last_cpu = measure_and_print("モデル読み込み", last_time, last_cpu, p)
        self.last_loop_time = time.time() # Watchdog reset

        self._setup_preprocessing(self.model)
        last_time, last_cpu = measure_and_print("前処理プロセス統合", last_time, last_cpu, p)
        self.last_loop_time = time.time() # Watchdog reset

        print(f"[YOLO] モデルをデバイス({self.device_name})向けにコンパイル中...")
        self.compiled_model = self.core.compile_model(self.model, self.device_name)
        last_time, last_cpu = measure_and_print("モデルコンパイル", last_time, last_cpu, p)
        self.last_loop_time = time.time() # Watchdog reset

        self.infer_queue = ov.AsyncInferQueue(self.compiled_model, jobs=0)
        self.infer_queue.set_callback(self._completion_callback)

        # --- 顔検出モデルの準備 (OpenVINO) ---
        face_model_dir = os.path.join(os.path.dirname(self.face_model_path), f'yolov12{self.face_model_size}-face_openvino_model')
        face_xml_path = os.path.join(face_model_dir, f'yolov12{self.face_model_size}-face.xml')

        if not os.path.exists(face_xml_path):
            if os.path.exists(self.face_model_path):
                print(f"[YOLO] 顔検出モデルをOpenVINO形式に変換しています: {self.face_model_path}")
                try:
                    self.last_loop_time = time.time() # Watchdog reset
                    model = YOLO(self.face_model_path)
                    model.export(format='openvino', half=True) # FP16でエクスポート
                    print("[YOLO] 変換完了")
                    self.last_loop_time = time.time() # Watchdog reset
                except Exception as e:
                    print(f"[YOLO] 顔検出モデルの変換に失敗: {e}")
            else:
                print(f"[YOLO] 顔検出モデル(.pt)が見つかりません: {self.face_model_path}")

        if os.path.exists(face_xml_path):
            try:
                print(f"[YOLO] 顔検出モデル(OpenVINO)を読み込んでいます: {face_xml_path}")
                self.face_net = self.core.read_model(face_xml_path)
                self._setup_preprocessing(self.face_net) # 同じ前処理を適用
                self.last_loop_time = time.time() # Watchdog reset
                
                print(f"[YOLO] 顔検出モデルをデバイス({self.device_name})向けにコンパイル中...")
                self.compiled_face_model = self.core.compile_model(self.face_net, self.device_name)
                self.face_infer_queue = ov.AsyncInferQueue(self.compiled_face_model, jobs=0)
                self.face_infer_queue.set_callback(self._face_completion_callback)
                print("[YOLO] 顔検出モデルの初期化完了")
                self.last_loop_time = time.time() # Watchdog reset
            except Exception as e:
                print(f"[YOLO] 顔検出モデル(OpenVINO)の初期化に失敗: {e}")
                traceback.print_exc()
        
        measure_and_print("初期化全体", total_start_time, total_start_cpu, p)
        self.is_ready_event.set() # 初期化完了を通知
        self.last_loop_time = time.time() # Watchdog reset

    def _setup_preprocessing(self, model):
        ppp = PrePostProcessor(model)
        input_layer = model.input(0)
        input_tensor_name = input_layer.any_name

        ppp.input(input_tensor_name).tensor() \
            .set_element_type(ov.Type.u8) \
            .set_layout(ov.Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR) \
            .set_spatial_dynamic_shape()

        ppp.input(input_tensor_name).preprocess() \
            .convert_element_type(ov.Type.f32) \
            .convert_color(ColorFormat.RGB) \
            .resize(ResizeAlgorithm.RESIZE_LINEAR) \
            .scale([255.0, 255.0, 255.0])

        ppp.input(input_tensor_name).model().set_layout(ov.Layout('NCHW'))
        model = ppp.build()

    def _should_reconnect_source(self, source):
        if not getattr(config, "AUTO_RECONNECT_CAMERA", True):
            return False
        if isinstance(source, int):
            return True
        if isinstance(source, str):
            src = source.strip().lower()
            if src.startswith(("rtsp://", "http://", "https://")):
                return True
        return False

    def _open_camera(self, source):
        cap = cv2.VideoCapture(source)
        self._configure_camera_capture(cap)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        return cap

    def _configure_camera_capture(self, cap):
        """カメラ入力の安定性向上のため、タイムアウト等の設定を試みる。"""
        if not cap:
            return
        read_timeout_ms = int(float(getattr(config, "CAMERA_READ_TIMEOUT_SECONDS", 2.0)) * 1000)
        # バッファを小さくして遅延を抑制
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # OpenCVのタイムアウト設定（対応バックエンドのみ有効）
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, read_timeout_ms)
            except Exception:
                pass
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms)
            except Exception:
                pass

    def _reconnect_camera(self, source):
        base_delay = float(getattr(config, "CAMERA_RECONNECT_BASE_DELAY_SECONDS", 0.5))
        max_delay = float(getattr(config, "CAMERA_RECONNECT_MAX_DELAY_SECONDS", 5.0))
        backoff = float(getattr(config, "CAMERA_RECONNECT_BACKOFF", 1.5))
        attempt = 0
        while not self.stop_event.is_set():
            self.last_loop_time = time.time() # Update watchdog
            attempt += 1
            delay = min(max_delay, base_delay * (backoff ** (attempt - 1)))
            print(f"[YOLO] Camera reconnect attempt {attempt}, waiting {delay:.1f}s...")
            Logger.log_system_event("INFO", "YOLO camera reconnect", message=f"attempt={attempt}, delay={delay:.1f}s")
            time.sleep(delay)
            self.last_loop_time = time.time() # Update watchdog
            cap = self._open_camera(source)
            if cap:
                print("[YOLO] Camera reconnected.")
                Logger.log_system_event("INFO", "YOLO camera reconnect", message="Reconnected successfully")
                return cap
        return None

    def _completion_callback(self, request, userdata):
        try:
            original_frame, start_time, h_scale, w_scale = userdata
            inference_end_time = time.time()
            process_time_ms = (inference_end_time - start_time) * 1000

            output_tensor = request.get_output_tensor(0).data
            if output_tensor.ndim == 3:
                output_tensor = output_tensor[0]

            detections = np.transpose(output_tensor)
            scores = np.max(detections[:, 4:], axis=1)
            mask = scores > config.CONFIDENCE_THRESHOLD
            
            filtered_detections = detections[mask]
            filtered_scores = scores[mask]

            results = []
            if len(filtered_detections) > 0:
                filtered_class_ids = np.argmax(filtered_detections[:, 4:], axis=1)
                
                cx = filtered_detections[:, 0]
                cy = filtered_detections[:, 1]
                w = filtered_detections[:, 2]
                h = filtered_detections[:, 3]

                x1 = (cx - w / 2) * w_scale
                y1 = (cy - h / 2) * h_scale
                w_pixel = w * w_scale
                h_pixel = h * h_scale

                boxes = np.stack([x1, y1, w_pixel, h_pixel], axis=1).astype(int).tolist()
                confidences = filtered_scores.tolist()
                class_ids = filtered_class_ids.tolist()

                indices = cv2.dnn.NMSBoxes(boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)

                if len(indices) > 0:
                    for i in indices.flatten():
                        results.append({
                            'box': boxes[i],
                            'confidence': confidences[i],
                            'class_id': class_ids[i]
                        })

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put((original_frame, results, process_time_ms))

        except Exception as e:
            Logger.log_system_error("YOLO 推論コールバック", e)
            print("Error in callback:")
            traceback.print_exc()

    def _face_completion_callback(self, request, userdata):
        try:
            original_frame, start_time, h_scale, w_scale = userdata
            
            output_tensor = request.get_output_tensor(0).data
            if output_tensor.ndim == 3:
                output_tensor = output_tensor[0]

            # YOLO output format: [x, y, w, h, conf, ...]
            detections = np.transpose(output_tensor)
            
            # 顔検出モデルは通常1クラスなので、スコアはindex 4にあると仮定
            # もしクラス確率が続くなら np.max(detections[:, 4:], axis=1) を使うが、
            # 1クラスなら detections[:, 4] がスコア
            if detections.shape[1] > 4:
                scores = detections[:, 4]
            else:
                scores = np.zeros(detections.shape[0])

            mask = scores > self.face_confidence_threshold
            
            filtered_detections = detections[mask]
            filtered_scores = scores[mask]

            results = []
            if len(filtered_detections) > 0:
                cx = filtered_detections[:, 0]
                cy = filtered_detections[:, 1]
                w = filtered_detections[:, 2]
                h = filtered_detections[:, 3]

                x1 = (cx - w / 2) * w_scale
                y1 = (cy - h / 2) * h_scale
                w_pixel = w * w_scale
                h_pixel = h * h_scale

                boxes = np.stack([x1, y1, w_pixel, h_pixel], axis=1).astype(int).tolist()
                confidences = filtered_scores.tolist()

                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.face_confidence_threshold, config.NMS_THRESHOLD)

                if len(indices) > 0:
                    for i in indices.flatten():
                        results.append({
                            'box': boxes[i],
                            'conf': confidences[i]
                        })

            if self.face_result_queue.full():
                try:
                    self.face_result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.face_result_queue.put(results)

        except Exception as e:
            Logger.log_system_error("YOLO 顔推論コールバック", e)
            print("Error in face callback:")
            traceback.print_exc()

    def set_gemini_response(self, text):
        with self.gemini_lock:
            self.gemini_response = text

    def set_voicevox_message(self, text):
        with self.gemini_lock:
            self.voicevox_message = text

    def _draw_gemini_response(self, frame):
        with self.gemini_lock:
            text = self.gemini_response
            
        if not text:
            return

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        alpha = 0.6
        margin = 20
        line_height = 30
        max_text_width = w - (margin * 2)
        
        if text == self.cached_gemini_text and self.cached_gemini_lines:
            lines = self.cached_gemini_lines
        else:
            lines = []
            paragraphs = text.split('\n')
            for paragraph in paragraphs:
                current_line = ""
                for char in paragraph:
                    test_line = current_line + char
                    (text_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                    if text_w <= max_text_width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = char
                if current_line:
                    lines.append(current_line)
            self.cached_gemini_text = text
            self.cached_gemini_lines = lines
                
        if not lines:
            return

        box_height = (len(lines) * line_height) + (margin * 2)
        start_y = h - box_height - 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, start_y), (w - margin, start_y + box_height), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        y = start_y + margin + 20
        for line in lines:
            cv2.putText(frame, line, (margin + 10, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height

    def _command_worker(self):
        """ロボット制御コマンドを送信するワーカースレッド"""
        while not self.stop_event.is_set():
            try:
                command_data = self.command_queue.get(timeout=0.5)
                url = f"http://127.0.0.1:{config.HTTP_SERVER_PORT}/pose"
                
                json_data = {}
                if isinstance(command_data, dict):
                    if 'body' in command_data:
                        json_data["CSotaMotion.SV_BODY_Y"] = int(command_data['body'])
                    if 'head' in command_data:
                        json_data["CSotaMotion.SV_HEAD_P"] = int(command_data['head'])
                else:
                    # Fallback for legacy or simple int
                    json_data["CSotaMotion.SV_BODY_Y"] = int(command_data)
                
                requests.post(url, json=json_data, timeout=0.1, proxies={"http": None, "https": None})
            except queue.Empty:
                continue
            except Exception as e:
                # 通信エラー等は無視するが、ConnectionResetErrorなどはログに残す
                # ただし、頻繁に出る可能性があるので、デバッグレベルにするか、
                # 特定のエラーだけログに残す
                # ここでは、ConnectionResetErrorがログに出ていたので、それをキャッチして
                # ログ出力を抑制するか、あるいはエラーとして記録するか検討する。
                # ユーザーのログには ConnectionResetError が出ていた。
                # これが原因でスレッドが止まることはないはず（try-except内なので）
                # しかし、念のためログに残しておく
                # Logger.log_system_error("Robot Command Error", e) 
                pass

    def _draw_results(self, frame, results, face_results, process_time_ms):
        PERSON_HEIGHT_THRESHOLD = 360
        current_time = time.time()
        detection_count = len(results)
        best_h = 0
        max_person_h = 0
        person_center_x = None

        # 追跡用変数
        frame_center_x = frame.shape[1] / 2
        frame_height = frame.shape[0]
        closest_person_cx = None
        closest_person_cy = None
        
        # ターゲット追跡ロジック (顔)
        matched_face = None
        
        # セッション維持判定用フラグ
        person_detected_for_session = False
        
        # 1. 既存のターゲットを追跡
        if self.target_box is not None:
            tx, ty, tw, th = self.target_box
            tcx = tx + tw / 2
            tcy = ty + th / 2
            
            best_match = None
            min_dist = float('inf')
            
            for res in face_results:
                x, y, w, h = res['box']
                cx = x + w / 2
                cy = y + h / 2
                
                # 中心点間の距離の二乗
                dist = (cx - tcx)**2 + (cy - tcy)**2
                if dist < min_dist:
                    min_dist = dist
                    best_match = res
            
            # 閾値判定 (例: 画面幅の1/3程度動いたらロストとするか、あるいは単に一番近いものを採用するか)
            # ここではある程度近い場合のみ同一人物とみなす
            # 1280x720なので、例えば400px以上飛んだら別人/誤検出の可能性
            if best_match and min_dist < (400**2):
                matched_face = best_match
                self.target_box = best_match['box']
            else:
                self.target_box = None # ターゲットロスト

        # 2. ターゲットがいない場合、新規ターゲットを選定 (中央に一番近い顔)
        if self.target_box is None and face_results:
            best_match = None
            min_dist_from_center = float('inf')
            
            for res in face_results:
                x, y, w, h = res['box']
                cx = x + w / 2
                dist = abs(cx - frame_center_x)
                
                if dist < min_dist_from_center:
                    min_dist_from_center = dist
                    best_match = res
            
            if best_match:
                matched_face = best_match
                self.target_box = best_match['box']

        # --- 顔検出の結果描画 ---
        for res in face_results:
            x, y, w, h = res['box']
            conf = res['conf']
            
            if res == matched_face:
                # ターゲット
                color = (0, 0, 255) # 赤
                text = f"Target {conf:.2f}"
                closest_person_cx = x + w / 2
                closest_person_cy = y + h / 2
            else:
                # その他の顔
                color = (0, 255, 0) # 緑
                text = f"Face {conf:.2f}"
            
            # 顔の描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- OpenVINO物体検出の結果描画 ---
        min_dist_from_center = float('inf') # Reset for body search fallback
        
        # 人数カウント用
        person_count = 0

        for res in results:
            x, y, w, h = res['box']
            confidence = res['confidence']
            class_id = res['class_id']
            label = config.CLASSES.get(class_id, 'Unknown')

            if label == 'person':
                person_count += 1
                if h > max_person_h:
                    max_person_h = h
                
                # セッション維持のための判定
                # 高さ 150px 以上、かつ 設定の信頼度しきい値以上
                if h > 150 and confidence >= config.CONFIDENCE_THRESHOLD:
                    person_detected_for_session = True

                # 通知用のロジック (既存)
                if h > PERSON_HEIGHT_THRESHOLD:
                    if h > best_h:
                        best_h = h
                        person_center_x = x + w / 2

                # 顔が検出されなかった場合のフォールバック追跡
                if closest_person_cx is None:
                    cx = x + w / 2
                    cy = y + h / 2
                    dist = abs(cx - frame_center_x)
                    if dist < min_dist_from_center:
                        min_dist_from_center = dist
                        closest_person_cx = cx
                        closest_person_cy = cy

            if label == 'person' and h > PERSON_HEIGHT_THRESHOLD:
                self.last_large_person_time = current_time
                # 通知済みフラグをチェック
                if not self.has_notified_in_session:
                    if (current_time - self.last_detection_time) > config.DETECTION_INTERVAL:
                        print(f" >> 通知: 大きな人物を検出しました (高さ: {h}px)")

                        # 修正: 先に人数を数える
                        current_person_count = sum(1 for r in results if config.CLASSES.get(r['class_id']) == 'person')

                        Logger.log_yolo_event(f"人物検出 (高さ: {h}px)", person_count=current_person_count)

                        should_update_time = True
                        if self.on_detection:
                            try:
                                # コールバックが False を返したら時間を更新しない（Gemini準備中など）
                                if self.on_detection() is False:
                                    should_update_time = False
                            except Exception as e:
                                Logger.log_system_error("YOLO on_detection callback", e)
                                print(f"[YOLO Error] on_detection callback failed: {e}")

                        if should_update_time:
                            self.last_detection_time = current_time

            color = (255, 0, 0) # 人物は青色で描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label_text = f"{label} {confidence:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(frame, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if person_detected_for_session:
            self.last_person_seen_time = current_time
        else:
            # 人が映っていない場合、last_person_seen_timeを更新しない
            # これにより、Gemini側で「最後に人を見た時間」からの経過時間を計測できる
            pass

        if self.pending_notification_reset:
            try:
                reset_after = float(getattr(config, "SESSION_TIMEOUT_SECONDS", 30))
                if self.last_large_person_time == 0.0 or (current_time - self.last_large_person_time) > reset_after:
                    with self.lock:
                        self.has_notified_in_session = False
                        self.pending_notification_reset = False
                        print("[YOLO] Notification flag reset (auto, after timeout).")
            except Exception as e:
                Logger.log_system_error("YOLO pending_notification_reset", e)
                print(f"[YOLO Error] Failed to reset pending notification: {e}")

        # Safety net for Low FPS mode
        # GeminiがクラッシュしてLow FPSモードが解除されなかった場合の対策
        if self.low_fps_mode and (current_time - self.last_detection_time > 60.0):
             print("[YOLO] Low FPS mode stuck? Forcing reset.")
             Logger.log_system_event("WARNING", "YOLO safety net", message="Forcing Low FPS mode OFF and resetting notification flag")
             self.set_low_fps_mode(False)
             self.reset_notification_flag()

        # ロボット追跡コマンドの送信
        if closest_person_cx is not None:
            width = frame.shape[1]
            height = frame.shape[0]
            
            # Body Y calculation
            raw_val = (closest_person_cx / width) * 1800 - 900
            body_y = -1 * raw_val 
            body_y = max(-900, min(900, body_y))
            
            # Head P calculation
            # Top (-290) <-> Center (0) <-> Bottom (80)
            head_p = 0
            if closest_person_cy < height / 2:
                # Upper half: Map [0, h/2] to [-290, 0]
                head_p = -290 + (closest_person_cy / (height / 2)) * 290
            else:
                # Lower half: Map [h/2, h] to [0, 80]
                head_p = ((closest_person_cy - height / 2) / (height / 2)) * 80
            
            head_p = max(-290, min(80, head_p))

            # キューに最新のコマンドを入れる（古いものは捨てる）
            if self.command_queue.full():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    pass
            self.command_queue.put({'body': body_y, 'head': head_p})

        self._draw_gemini_response(frame)

        self.frame_count += 1
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
            self.inference_time_ms = process_time_ms

        panel_h, panel_w = 160, 240
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        info_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_y = 35

        cv2.putText(frame, f"YOLOv12 OpenVINO", (20, start_y), font, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Device: {self.device_name}", (20, start_y + 25), font, 0.5, info_color, 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, start_y + 50), font, 0.5, info_color, 1)
        cv2.putText(frame, f"Latency: {self.inference_time_ms:.1f} ms", (20, start_y + 75), font, 0.5, info_color, 1)
        cv2.putText(frame, f"Objects: {detection_count}", (20, start_y + 100), font, 0.5, (0, 255, 0), 1)
        
        if self.voicevox_message:
            cv2.putText(frame, f"Voicevox: {self.voicevox_message}", (20, start_y + 125), font, 0.5, (0, 255, 255), 1)

        return frame

    def _watchdog_worker(self):
        """YOLOスレッドの生存監視"""
        Logger.log_system_event("INFO", "YOLO Watchdog", message="Started")
        print("[YOLO] Watchdog started.")
        while not self.stop_event.is_set():
            try:
                time.sleep(5)
                if self.pause_event.is_set():
                    self.last_loop_time = time.time() # Prevent timeout during pause
                    continue
                
                elapsed = time.time() - self.last_loop_time
                if elapsed > 30.0: # 30秒以上更新がない
                    msg = f"YOLO thread stalled for {elapsed:.1f}s (FPS: {self.fps:.1f})"
                    print(f"[YOLO Watchdog] {msg}")
                    Logger.log_system_error("YOLO Watchdog", message=msg)
                    
                    # 自動再起動を試みる
                    print("[YOLO Watchdog] Attempting to restart YOLO thread...")
                    self.restart()
                    # 再起動後はループを抜ける（新しいWatchdogが起動されるため）
                    break
            except Exception as e:
                Logger.log_system_error("YOLO Watchdog Error", e)
                print(f"[YOLO Watchdog] Error: {e}")
                time.sleep(5) # エラー発生時は少し待つ

    def run(self, source=None):
        show_window = False
        try:
            self.last_loop_time = time.time() # Update start time
            self._initialize_dependencies()
            self.last_loop_time = time.time() # Update after init

            if source is None:
                source = config.VIDEO_SOURCE

            Logger.log_system_event("INFO", "YOLO run start", message=f"Source: {source}")

            show_window = bool(getattr(config, "SHOW_OPENCV_WINDOW", True))
            if show_window and threading.current_thread() is not threading.main_thread():
                allow_in_thread = bool(getattr(config, "ALLOW_OPENCV_WINDOW_IN_THREAD", True))
                if not allow_in_thread:
                    show_window = False
                    print("[YOLO] OpenCV window disabled (not running in main thread).")
                    Logger.log_system_event("INFO", "YOLO viewer", message="OpenCV window disabled (not in main thread)")
                else:
                    print("[YOLO] OpenCV window enabled in worker thread (ALLOW_OPENCV_WINDOW_IN_THREAD=True).")
                    Logger.log_system_event("INFO", "YOLO viewer", message="OpenCV window allowed in worker thread")

            cap = self._open_camera(source)
            self.last_loop_time = time.time() # Update after camera open

            if not cap:
                if self._should_reconnect_source(source):
                    print("[YOLO] Camera open failed. Attempting to reconnect...")
                    Logger.log_system_event("INFO", "YOLO camera reconnect", message="Initial open failed, starting reconnect loop")
                    cap = self._reconnect_camera(source)
                    self.last_loop_time = time.time() # Update after reconnect
                if not cap:
                    print("[エラー] カメラを開けませんでした。")
                    return

            print("\n[開始] 推論ループを開始します。")

            last_heartbeat_time = time.time()
            heartbeat_interval = 30.0  # 30秒ごとにハートビート記録
            loop_iteration = 0

            while not self.stop_event.is_set():
                try:
                    loop_iteration += 1
                    loop_start_time = time.time() # ループ開始時間
                    self.last_loop_time = loop_start_time # 外部監視用

                    # ハートビート記録
                    if loop_start_time - last_heartbeat_time >= heartbeat_interval:
                        # メモリ使用量を取得
                        import psutil
                        process = psutil.Process()
                        mem_info = process.memory_info()
                        mem_mb = mem_info.rss / 1024 / 1024

                        print(f"[YOLO] Heartbeat: Loop running (FPS: {self.fps:.1f}, iteration: {loop_iteration}, memory: {mem_mb:.1f}MB)")
                        Logger.log_system_event("INFO", "YOLO heartbeat",
                            message=f"Loop iteration {loop_iteration}, FPS {self.fps:.1f}, stop_event={self.stop_event.is_set()}, pause_event={self.pause_event.is_set()}, memory_mb={mem_mb:.1f}")
                        last_heartbeat_time = loop_start_time

                    if self.pause_event.is_set():
                        time.sleep(0.1)
                        continue

                    read_timeout_s = float(getattr(config, "CAMERA_READ_TIMEOUT_SECONDS", 2.0))
                    read_start = time.monotonic()
                    ret, frame = cap.read()
                    read_elapsed = time.monotonic() - read_start
                    read_timed_out = read_timeout_s > 0 and read_elapsed > read_timeout_s
                    if read_timed_out:
                        message = (
                            f"カメラ読み込みがタイムアウトしました "
                            f"(elapsed={read_elapsed:.2f}s, timeout={read_timeout_s:.2f}s, iteration={loop_iteration})"
                        )
                        # 連続ログ抑制
                        now = time.time()
                        if now - self._last_camera_timeout_log > 5.0:
                            Logger.log_system_error("YOLO カメラストリーム", message=message)
                            self._last_camera_timeout_log = now
                        print(f"[情報] {message}")
                        ret = False

                    if not ret:
                        if not read_timed_out:
                            message = f"映像ストリーム終了 (カメラ切断またはエラー) at iteration {loop_iteration}"
                            print(f"[情報] {message}")
                            Logger.log_system_error("YOLO カメラストリーム", message=message)
                        if self._should_reconnect_source(source):
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = self._reconnect_camera(source)
                            self.last_loop_time = time.time() # Update after reconnect
                            if cap:
                                continue
                        break

                    self.last_frame_time = time.time()
                    self.current_frame_for_capture = frame.copy()

                    h, w = frame.shape[:2]
                    h_scale = h / self.input_height
                    w_scale = w / self.input_width

                    input_tensor = np.expand_dims(frame, 0)

                    # メインモデルの推論開始
                    self.infer_queue.start_async(
                        inputs={0: input_tensor},
                        userdata=(frame, time.time(), h_scale, w_scale)
                    )

                    # 顔検出モデルの推論開始 (OpenVINO)
                    if self.face_infer_queue:
                        self.face_infer_queue.start_async(
                            inputs={0: input_tensor},
                            userdata=(frame, time.time(), h_scale, w_scale)
                        )

                    # 結果の取得 (メイン)
                    try:
                        while True:
                            _, r_results, r_time = self.result_queue.get_nowait()
                            self.latest_results = r_results
                            self.inference_time_ms = r_time
                    except queue.Empty:
                        pass

                    # 結果の取得 (顔)
                    try:
                        while True:
                            f_results = self.face_result_queue.get_nowait()
                            self.latest_face_results = f_results
                    except queue.Empty:
                        pass

                    display_frame = frame.copy()
                    self._draw_results(display_frame, self.latest_results, self.latest_face_results, self.inference_time_ms)

                    if show_window:
                        try:
                            cv2.imshow("YOLOv12 - Visualized Detection", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("[終了] ユーザー指示")
                                break
                        except Exception as cv_err:
                            # cv2.waitKey/imshowがハングまたは例外を起こした場合
                            Logger.log_system_error("YOLO OpenCV表示", cv_err)
                            print(f"[YOLO Warning] OpenCV display error: {cv_err}")
                            # 継続して処理を続ける（表示なしで）

                    # FPS制御
                    elapsed_time = time.time() - loop_start_time
                    target_frame_time = 1.0 / self.target_fps
                    if elapsed_time < target_frame_time:
                        time.sleep(target_frame_time - elapsed_time)

                except Exception as loop_err:
                    # ループ内での予期しないエラーをキャッチ
                    Logger.log_system_error("YOLO ループ内エラー", loop_err)
                    print(f"[YOLO Error] Loop error: {loop_err}")
                    traceback.print_exc()
                    # ループを継続（致命的でない限り）
                    time.sleep(0.1)

        except BaseException as e:
            Logger.log_system_error("YOLO 実行 (BaseException)", e)
            print(f"[YOLO Critical Error] {e}")
            traceback.print_exc()
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
        finally:
            # ループ終了理由を記録
            exit_reason = "unknown"
            if 'loop_iteration' in locals():
                exit_reason = f"Loop exited at iteration {loop_iteration}"
            if self.stop_event.is_set():
                exit_reason += ", stop_event was set"
            if 'ret' in locals() and not ret:
                exit_reason += ", camera stream ended"

            print(f"[待機] 推論タスク完了待ち... ({exit_reason})")
            Logger.log_system_error("YOLO thread終了",
                message=f"YOLO thread is shutting down. Reason: {exit_reason}, stop_event={self.stop_event.is_set()}, loop_iteration={locals().get('loop_iteration', 'N/A')}")

            if self.infer_queue:
                # self.infer_queue.wait_all() # <--- ここがハングの原因の可能性があるためコメントアウト
                pass
            if self.face_infer_queue:
                # self.face_infer_queue.wait_all() # <--- ここも同様
                pass
            if 'cap' in locals() and cap:
                cap.release()
            if cv2 and show_window:
                cv2.destroyAllWindows()
            print(f"[YOLO] 終了しました ({exit_reason})")
            Logger.log_system_event("INFO", "YOLO lifecycle", message=f"YOLO thread terminated: {exit_reason}")

    def get_current_frame(self):
        return self.current_frame_for_capture

    def pause(self):
        if not self.pause_event.is_set():
            print("[YOLO] 一時停止します。")
            self.pause_event.set()

    def resume(self):
        if self.pause_event.is_set():
            print("[YOLO] 再開します。")
            self.pause_event.clear()

    def start(self):
        thread_alive = self.thread and self.thread.is_alive()
        if thread_alive:
            return

        self.stop_event.clear()

        # ロボット制御用スレッドの開始
        if not self.command_thread or not self.command_thread.is_alive():
            self.command_thread = threading.Thread(target=self._command_worker, daemon=True)
            self.command_thread.start()

        # Watchdogの開始
        if not self.watchdog_thread or not self.watchdog_thread.is_alive():
            self.watchdog_thread = threading.Thread(target=self._watchdog_worker, daemon=True)
            self.watchdog_thread.start()

        # YOLO検出スレッドの開始
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        print("[YOLO] スレッドを開始しました。")

    def stop(self):
        if self.thread and self.thread.is_alive():
            print("[YOLO] 停止シグナルを送信します...")
            self.stop_event.set()
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                print("[YOLO] スレッドの停止に失敗しました。")
            else:
                print("[YOLO] スレッドが正常に停止しました。")
        
        # ロボット制御用スレッドの停止
        if hasattr(self, 'command_thread') and self.command_thread and self.command_thread.is_alive():
             self.command_thread.join(timeout=1)
        
        # Watchdogの停止
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=1)
        self.watchdog_thread = None

        self.thread = None
        self.command_thread = None

    def restart(self):
        """スレッドを強制的に停止して再起動する"""
        print("[YOLO] Restarting YOLO thread...")
        Logger.log_system_event("INFO", "YOLO lifecycle", message="Restart requested")
        self.stop()
        # 強制的にNoneにして再作成を促す
        self.thread = None
        time.sleep(1.0)
        self.start()

def simple_callback():
    pass

def main():
    if not os.path.exists(config.MODEL_PATH):
        print(f"[エラー] {config.MODEL_PATH} がありません")
        return

    try:
        optimizer = YOLOOptimizer(
            on_detection=simple_callback
        )
        optimizer.start()
        input("Enterキーを押すと終了します...\n")
    except Exception as e:
        print(f"[致命的エラー]:\n{e}")
        traceback.print_exc()
    finally:
        if 'optimizer' in locals() and optimizer:
            optimizer.stop()

if __name__ == "__main__":
    main()
