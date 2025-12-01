import cv2
import numpy as np
import openvino as ov
import time
import threading

class YOLODetector:
    def __init__(self, on_detection):
        """
        YOLO検出器を初期化します。
        :param on_detection: 検出時に呼び出されるコールバック関数
        """
        self.on_detection = on_detection
        self.running = False
        self.thread = None
        self.last_detection_time = 0
        self.detection_interval = 180  # 3分間のクールダウン

        # ---------------------------------------------------------
        # 1. OpenVINOランタイムの初期化
        # ---------------------------------------------------------
        print("OpenVINO Runtimeを初期化中...")
        core = ov.Core()
        model_xml_path = "yolov12n_openvino_model/yolov12n.xml"
        print(f"モデルをIntel GPUにロードしています... ({model_xml_path})")
        try:
            self.compiled_model = core.compile_model(model_xml_path, "GPU")
        except Exception as e:
            print(f"エラー: モデルの読み込みに失敗しました。\n{e}")
            raise

        self.infer_request = self.compiled_model.create_infer_request()

        # クラス名リスト
        self.classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }

        # ---------------------------------------------------------
        # 2. カメラ設定
        # ---------------------------------------------------------
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("カメラが開けませんでした。")
            raise IOError("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"カメラ解像度を {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} に設定しようと試みました。")

    def start(self):
        """検出ループをバックグラウンドスレッドで開始します。"""
        if self.running:
            print("YOLO detector is already running.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("YOLO detector started.")

    def stop(self):
        """検出ループを停止します。"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("YOLO detector stopped.")

    def _run_loop(self):
        """メインの検出ループ。"""
        print("=== Intel GPU リアルタイム検知 (OpenVINO Native) ===")
        print("終了するには 'q' キーを押してください。")

        PERSON_HEIGHT_THRESHOLD = 360

        try:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    break

                input_tensor = self._preprocess_frame(frame)
                detections = self._run_inference(input_tensor)
                self._postprocess_and_draw(frame, detections, PERSON_HEIGHT_THRESHOLD)

                cv2.imshow("YOLOv12 - Intel GPU Native", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False # 'q'キーでループを抜ける
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.running = False # Ensure flag is set
            print("YOLO detector loop finished.")

    def _preprocess_frame(self, frame):
        input_img = cv2.resize(frame, (640, 640))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_tensor = input_img.transpose(2, 0, 1)
        input_tensor = input_tensor[np.newaxis, :, :, :]
        input_tensor = input_tensor.astype(np.float32) / 255.0
        return input_tensor

    def _run_inference(self, input_tensor):
        results = self.infer_request.infer([input_tensor])
        detections = list(results.values())[0]
        return np.transpose(detections[0])

    def _postprocess_and_draw(self, frame, detections, threshold):
        boxes, confidences, class_ids = [], [], []
        h_scale = frame.shape[0] / 640
        w_scale = frame.shape[1] / 640

        scores = np.max(detections[:, 4:], axis=1)
        mask = scores > 0.5
        filtered_detections = detections[mask]

        for det in filtered_detections:
            class_id = np.argmax(det[4:])
            confidence = det[4 + class_id]
            cx, cy, w, h = det[0], det[1], det[2], det[3]
            x1 = int((cx - w / 2) * w_scale)
            y1 = int((cy - h / 2) * h_scale)
            width = int(w * w_scale)
            height = int(h * h_scale)
            boxes.append([x1, y1, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = self.classes.get(class_ids[i], 'Unknown')
                conf = confidences[i]

                current_time = time.time()
                if label == 'person' and h > threshold and (current_time - self.last_detection_time) > self.detection_interval:
                    print("しきい値に達しました。コールバックを呼び出します。")
                    if self.on_detection:
                        self.on_detection()
                    self.last_detection_time = current_time

                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    """スタンドアロン実行用のテスト関数"""
    def simple_callback():
        print("--- 検出イベント受信！ ---")

    detector = None
    try:
        detector = YOLODetector(on_detection=simple_callback)
        # このデモではメインスレッドで実行し、ウィンドウが閉じるのを待ちます
        detector._run_loop()
    except Exception as e:
        print(f"実行中にエラーが発生しました: {e}")
    finally:
        if detector:
            detector.stop()

if __name__ == '__main__':
    main()