import cv2
import time
import numpy as np
import threading
import os
import sys
import queue
import traceback # エラー詳細表示用
from pathlib import Path

# OpenVINOのインポート
import openvino as ov
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat

# ==========================================
# 設定パラメータ (Configuration)
# ==========================================
MODEL_PATH = "yolov12n_openvino_model/yolov12n.xml"
# 修正: Intel GPUを明示的に使用するため "GPU" に変更しました
# (もしGPUがない環境でエラーになる場合は "AUTO" または "CPU" に戻してください)
DEVICE_NAME = "GPU"
CACHE_DIR = "./model_cache"
MODEL_INPUT_SIZE = (640, 640)

# 検出パラメータ
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
DETECTION_INTERVAL = 180.0

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

class YOLOOptimizer:
    def __init__(self, model_path, device_name, cache_dir, input_size, on_detection=None):
        """
        YOLO検出エンジンの初期化
        """
        self.on_detection = on_detection
        self.input_width, self.input_height = input_size
        self.last_detection_time = 0
        self.lock = threading.Lock()
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.inference_time_ms = 0.0

        # 結果受け渡し用キュー (最大サイズを小さくして遅延を防ぐ)
        self.result_queue = queue.Queue(maxsize=2)

        if not os.path.exists(model_path):
            print(f"\n[エラー] モデルファイルが見つかりません: {os.path.abspath(model_path)}")
            print("以下のコマンドを実行してモデルをエクスポートしてください:")
            print("  yolo export model=yolov12n.pt format=openvino\n")
            raise FileNotFoundError(f"Model not found at {model_path}")

        print("[初期化] OpenVINO Runtimeを起動中...")
        self.core = ov.Core()

        # ログ追加: 認識されているデバイス一覧を表示
        print(f"[情報] 利用可能なOpenVINOデバイス: {self.core.available_devices}")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.core.set_property({ov.properties.cache_dir: str(cache_path)})
        print(f"[情報] モデルキャッシュを有効化: {cache_path}")

        # GPU利用時の最適化設定
        if "GPU" in device_name or "AUTO" in device_name:
            self.core.set_property(device_name, {
                ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.THROUGHPUT
            })

        print(f"[情報] モデルを読み込んでいます: {model_path}")
        self.model = self.core.read_model(model_path)

        self._setup_preprocessing()

        print(f"[情報] モデルをデバイス({device_name})向けにコンパイル中...")
        try:
            self.compiled_model = self.core.compile_model(self.model, device_name)
        except Exception as e:
            print(f"[エラー] モデルのコンパイルに失敗しました。\n詳細: {e}")
            print("ヒント: GPUドライバが古い場合や、デバイス名が異なる可能性があります。")
            raise

        self.infer_queue = ov.AsyncInferQueue(self.compiled_model, jobs=0)
        self.infer_queue.set_callback(self._completion_callback)

        num_jobs = len(self.infer_queue) if hasattr(self.infer_queue, '__len__') else self.infer_queue.jobs_number
        print(f"[情報] 非同期推論キューを初期化完了 (並列ジョブ数: {num_jobs})")


    def _setup_preprocessing(self):
        """
        モデルに入力する前の画像処理定義
        """
        ppp = PrePostProcessor(self.model)
        input_layer = self.model.input(0)
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

        print("[情報] 前処理プロセスをモデルに統合しました")
        self.model = ppp.build()

    def _completion_callback(self, request, userdata):
        """
        推論完了コールバック (別スレッド)
        注意: ここでは描画処理(cv2.rectangle等)は行いません。計算結果のみを返します。
        """
        try:
            original_frame, start_time, h_scale, w_scale = userdata

            inference_end_time = time.time()
            process_time_ms = (inference_end_time - start_time) * 1000

            output_tensor = request.get_output_tensor(0).data

            # --- 出力テンソルの形状調整 ---
            # output_tensorの形状は通常 (1, 84, 8400) = (Batch, 4+Classes, Anchors)
            # これを (8400, 84) = (Anchors, 4+Classes) の2次元配列に直す必要があります

            # 1. バッチ次元を削除: (1, 84, 8400) -> (84, 8400)
            if output_tensor.ndim == 3:
                output_tensor = output_tensor[0]

            # 2. 転置して (行=アンカー, 列=属性) の形にする: (84, 8400) -> (8400, 84)
            detections = np.transpose(output_tensor)

            # --- NumPyを使った高速フィルタリング ---

            # 1. 各ボックスの最大クラススコアを取得 (4列目以降がクラススコア)
            # detections[:, 4:] は (8400, 80)
            scores = np.max(detections[:, 4:], axis=1) # -> (8400,) の1次元配列

            # 2. しきい値(CONFIDENCE_THRESHOLD)以下をカット
            mask = scores > CONFIDENCE_THRESHOLD # -> (8400,) の1次元ブール配列

            # detections (8400, 84) に対し mask (8400,) で行を抽出
            filtered_detections = detections[mask]
            filtered_scores = scores[mask]

            results = [] # 検出結果のリスト

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

                # NMS
                indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

                if len(indices) > 0:
                    for i in indices.flatten():
                        # 結果を辞書形式で保存
                        results.append({
                            'box': boxes[i],
                            'confidence': confidences[i],
                            'class_id': class_ids[i]
                        })

            # 計算結果をメインスレッドへ送信
            # キューが一杯なら古いものを捨てて最新を入れる
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass

            self.result_queue.put((original_frame, results, process_time_ms))

        except Exception:
            # コールバック内でのエラーはキャッチしないとプロセスごと落ちる(0xC0000409)
            print("Error in callback:")
            traceback.print_exc()

    def _draw_results(self, frame, results, process_time_ms):
        """
        描画処理 (メインスレッドで実行)
        """
        PERSON_HEIGHT_THRESHOLD = 360
        current_time = time.time()
        detection_count = len(results)

        # 物体の描画
        for res in results:
            x, y, w, h = res['box']
            confidence = res['confidence']
            class_id = res['class_id']
            label = CLASSES.get(class_id, 'Unknown')

            # 通知ロジック
            if label == 'person' and h > PERSON_HEIGHT_THRESHOLD:
                if (current_time - self.last_detection_time) > DETECTION_INTERVAL:
                    print(f" >> 通知: 大きな人物を検出しました (高さ: {h}px)")
                    if self.on_detection:
                        self.on_detection()
                    self.last_detection_time = current_time

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label_text = f"{label} {confidence:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(frame, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # パネル情報更新
        self.frame_count += 1
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
            self.inference_time_ms = process_time_ms

        # パネル描画
        panel_h, panel_w = 110, 240
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        info_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_y = 35

        cv2.putText(frame, f"YOLOv12 OpenVINO", (20, start_y), font, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Device: {DEVICE_NAME}", (20, start_y + 25), font, 0.5, info_color, 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, start_y + 50), font, 0.5, info_color, 1)
        cv2.putText(frame, f"Latency: {self.inference_time_ms:.1f} ms", (20, start_y + 75), font, 0.5, info_color, 1)
        cv2.putText(frame, f"Objects: {detection_count}", (20, start_y + 100), font, 0.5, (0, 255, 0), 1)

        return frame

    def run(self, source=0):
        """
        メイン実行ループ
        """
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[エラー] カメラを開けませんでした。")
            return

        print("\n[開始] 推論ループを開始します。")
        print("  - 'q' キー: 終了")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[情報] 映像ストリーム終了")
                    break

                h, w = frame.shape[:2]
                h_scale = h / self.input_height
                w_scale = w / self.input_width

                input_tensor = np.expand_dims(frame, 0)

                self.infer_queue.start_async(
                    inputs={0: input_tensor},
                    userdata=(frame, time.time(), h_scale, w_scale)
                )

                # --- メインスレッドでの描画と表示 ---
                try:
                    # 結果の取得（ブロックせず、なければスキップ）
                    result_data = self.result_queue.get_nowait()
                    if result_data:
                        r_frame, r_results, r_time = result_data

                        # ここで描画を行う (安全)
                        display_frame = self._draw_results(r_frame, r_results, r_time)

                        cv2.imshow("YOLOv12 - Visualized Detection", display_frame)

                except queue.Empty:
                    pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[終了] ユーザー指示")
                    break

        except KeyboardInterrupt:
            print("\n[終了] Ctrl+C")
        finally:
            print("[待機] 推論タスク完了待ち...")
            self.infer_queue.wait_all()
            cap.release()
            cv2.destroyAllWindows()
            print("[完了] 終了しました")

def simple_callback():
    pass

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[エラー] {MODEL_PATH} がありません")
        return

    try:
        optimizer = YOLOOptimizer(
            model_path=MODEL_PATH,
            device_name=DEVICE_NAME,
            cache_dir=CACHE_DIR,
            input_size=MODEL_INPUT_SIZE,
            on_detection=simple_callback
        )
        optimizer.run()
    except Exception as e:
        print(f"[致命的エラー]:\n{e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()