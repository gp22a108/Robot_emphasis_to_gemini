
import time
import threading
import os
import sys
import queue
import traceback
from pathlib import Path
import requests
import urllib.request

# 設定ファイルをインポート
import config

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
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.current_frame_for_capture = None
        self.last_person_seen_time = 0.0
        self.is_ready_event = threading.Event() # 初期化完了フラグ
        
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

    def is_ready(self):
        """初期化が完了したかどうかを返す"""
        return self.is_ready_event.is_set()

    def set_low_fps_mode(self, enabled):
        """低FPSモードの切り替え"""
        self.low_fps_mode = enabled
        if enabled:
            self.target_fps = 3.0
            print("[YOLO] Low FPS Mode: ON (3 FPS)")
        else:
            self.target_fps = 30.0
            print("[YOLO] Low FPS Mode: OFF (30 FPS)")

    def _initialize_dependencies(self):
        """依存ライブラリとモデルの初期化（別スレッドで実行）"""
        global cv2, np, ov, PrePostProcessor, ResizeAlgorithm, ColorFormat, psutil, YOLO
        
        if cv2 is not None:
            return

        print("[YOLO] ライブラリを読み込んでいます...")
        t_start = time.perf_counter()
        
        import cv2
        import numpy as np
        import openvino as ov
        from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
        import psutil
        from ultralytics import YOLO
        
        print(f"[YOLO] ライブラリ読み込み完了: {time.perf_counter() - t_start:.3f}s")

        # --- 計測の準備 ---
        p = psutil.Process()
        p.cpu_percent(None)
        total_start_time = time.perf_counter()
        total_start_cpu = time.process_time()
        last_time, last_cpu = total_start_time, total_start_cpu

        print("[YOLO] OpenVINO Runtimeを起動中...")
        self.core = ov.Core()
        last_time, last_cpu = measure_and_print("OpenVINO Runtime起動", last_time, last_cpu, p)

        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.core.set_property({ov.properties.cache_dir: str(cache_path)})
        
        if "GPU" in self.device_name or "AUTO" in self.device_name:
            self.core.set_property(self.device_name, {
                ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.THROUGHPUT
            })

        # --- メインモデルの読み込み ---
        if not os.path.exists(self.model_path):
            print(f"\n[エラー] モデルファイルが見つかりません: {os.path.abspath(self.model_path)}")
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"[YOLO] モデルを読み込んでいます: {self.model_path}")
        self.model = self.core.read_model(self.model_path)
        last_time, last_cpu = measure_and_print("モデル読み込み", last_time, last_cpu, p)

        self._setup_preprocessing(self.model)
        last_time, last_cpu = measure_and_print("前処理プロセス統合", last_time, last_cpu, p)

        print(f"[YOLO] モデルをデバイス({self.device_name})向けにコンパイル中...")
        self.compiled_model = self.core.compile_model(self.model, self.device_name)
        last_time, last_cpu = measure_and_print("モデルコンパイル", last_time, last_cpu, p)

        self.infer_queue = ov.AsyncInferQueue(self.compiled_model, jobs=0)
        self.infer_queue.set_callback(self._completion_callback)

        # --- 顔検出モデルの準備 (OpenVINO) ---
        face_model_dir = os.path.join(os.path.dirname(self.face_model_path), f'yolov12{self.face_model_size}-face_openvino_model')
        face_xml_path = os.path.join(face_model_dir, f'yolov12{self.face_model_size}-face.xml')

        if not os.path.exists(face_xml_path):
            if os.path.exists(self.face_model_path):
                print(f"[YOLO] 顔検出モデルをOpenVINO形式に変換しています: {self.face_model_path}")
                try:
                    model = YOLO(self.face_model_path)
                    model.export(format='openvino', half=True) # FP16でエクスポート
                    print("[YOLO] 変換完了")
                except Exception as e:
                    print(f"[YOLO] 顔検出モデルの変換に失敗: {e}")
            else:
                print(f"[YOLO] 顔検出モデル(.pt)が見つかりません: {self.face_model_path}")

        if os.path.exists(face_xml_path):
            try:
                print(f"[YOLO] 顔検出モデル(OpenVINO)を読み込んでいます: {face_xml_path}")
                self.face_net = self.core.read_model(face_xml_path)
                self._setup_preprocessing(self.face_net) # 同じ前処理を適用
                
                print(f"[YOLO] 顔検出モデルをデバイス({self.device_name})向けにコンパイル中...")
                self.compiled_face_model = self.core.compile_model(self.face_net, self.device_name)
                self.face_infer_queue = ov.AsyncInferQueue(self.compiled_face_model, jobs=0)
                self.face_infer_queue.set_callback(self._face_completion_callback)
                print("[YOLO] 顔検出モデルの初期化完了")
            except Exception as e:
                print(f"[YOLO] 顔検出モデル(OpenVINO)の初期化に失敗: {e}")
                traceback.print_exc()
        
        measure_and_print("初期化全体", total_start_time, total_start_cpu, p)
        self.is_ready_event.set() # 初期化完了を通知

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

        except Exception:
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

        except Exception:
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
                body_y = self.command_queue.get(timeout=0.5)
                url = f"http://127.0.0.1:{config.HTTP_SERVER_PORT}/pose"
                data = {"CSotaMotion.SV_BODY_Y": int(body_y)}
                requests.post(url, json=data, timeout=0.1)
            except queue.Empty:
                continue
            except Exception:
                # 通信エラー等は無視
                pass

    def _draw_results(self, frame, results, face_results, process_time_ms):
        PERSON_HEIGHT_THRESHOLD = 360
        current_time = time.time()
        detection_count = len(results)
        best_h = 0
        person_center_x = None

        # 追跡用変数
        frame_center_x = frame.shape[1] / 2
        closest_person_cx = None
        min_dist_from_center = float('inf')

        # --- 顔検出の結果描画 ---
        for res in face_results:
            x, y, w, h = res['box']
            conf = res['conf']
            
            # 顔の描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 追跡ロジック (顔優先)
            cx = x + w / 2
            dist = abs(cx - frame_center_x)
            if dist < min_dist_from_center:
                min_dist_from_center = dist
                closest_person_cx = cx

        # --- OpenVINO物体検出の結果描画 ---
        for res in results:
            x, y, w, h = res['box']
            confidence = res['confidence']
            class_id = res['class_id']
            label = config.CLASSES.get(class_id, 'Unknown')

            if label == 'person':
                # 通知用のロジック (既存)
                if h > PERSON_HEIGHT_THRESHOLD:
                    if h > best_h:
                        best_h = h
                        person_center_x = x + w / 2
                
                # 顔が検出されなかった場合のフォールバック追跡
                if closest_person_cx is None:
                    cx = x + w / 2
                    dist = abs(cx - frame_center_x)
                    if dist < min_dist_from_center:
                        min_dist_from_center = dist
                        closest_person_cx = cx

            if label == 'person' and h > PERSON_HEIGHT_THRESHOLD:
                if (current_time - self.last_detection_time) > config.DETECTION_INTERVAL:
                    print(f" >> 通知: 大きな人物を検出しました (高さ: {h}px)")
                    if self.on_detection:
                        self.on_detection()
                    self.last_detection_time = current_time

            color = (255, 0, 0) # 人物は青色で描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label_text = f"{label} {confidence:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(frame, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if best_h > 0 and person_center_x is not None:
            self.last_person_seen_time = current_time

        # ロボット追跡コマンドの送信
        if closest_person_cx is not None:
            width = frame.shape[1]
            # 画面中央(width/2) -> 0
            # 画面左端(0) -> 900 (変更)
            # 画面右端(width) -> -900 (変更)
            # 計算式: -1 * ((cx / width) * 1800 - 900)
            # または: 900 - (cx / width) * 1800
            
            raw_val = (closest_person_cx / width) * 1800 - 900
            body_y = -1 * raw_val # 符号を反転
            
            body_y = max(-900, min(900, body_y))
            
            # キューに最新のコマンドを入れる（古いものは捨てる）
            if self.command_queue.full():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    pass
            self.command_queue.put(body_y)

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

    def run(self, source=0):
        try:
            self._initialize_dependencies()
            
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not cap.isOpened():
                print("[エラー] カメラを開けませんでした。")
                return

            print("\n[開始] 推論ループを開始します。")
            
            while not self.stop_event.is_set():
                loop_start_time = time.time() # ループ開始時間

                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("[情報] 映像ストリーム終了")
                    break

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
                cv2.imshow("YOLOv12 - Visualized Detection", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[終了] ユーザー指示")
                    break
                
                # FPS制御
                elapsed_time = time.time() - loop_start_time
                target_frame_time = 1.0 / self.target_fps
                if elapsed_time < target_frame_time:
                    time.sleep(target_frame_time - elapsed_time)

        except Exception as e:
            print(f"[YOLO Error] {e}")
            traceback.print_exc()
        finally:
            print("[待機] 推論タスク完了待ち...")
            if self.infer_queue:
                self.infer_queue.wait_all()
            if self.face_infer_queue:
                self.face_infer_queue.wait_all()
            if 'cap' in locals() and cap:
                cap.release()
            if cv2:
                cv2.destroyAllWindows()
            print("[YOLO] 終了しました")

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
        if self.thread is None:
            self.stop_event.clear()
            
            # ロボット制御用スレッドの開始
            self.command_thread = threading.Thread(target=self._command_worker, daemon=True)
            self.command_thread.start()
            
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
             
        self.thread = None
        self.command_thread = None

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
