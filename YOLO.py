import cv2
import numpy as np
import openvino as ov

def main():
    # ---------------------------------------------------------
    # 1. OpenVINOランタイムの初期化
    # ---------------------------------------------------------
    print("OpenVINO Runtimeを初期化中...")
    core = ov.Core()

    # エクスポートされたモデルのパス（.xmlファイル）を指定
    # ※前回の実行でフォルダができているはずです
    model_xml_path = "yolov12n_openvino_model/yolov12n.xml"

    print(f"モデルをIntel GPUにロードしています... ({model_xml_path})")
    try:
        # ここで "GPU" を指定することで強制的にIntel GPUを使用します
        compiled_model = core.compile_model(model_xml_path, "GPU")
    except Exception as e:
        print(f"エラー: モデルの読み込みに失敗しました。\n{e}")
        print("詳細: もしかするとパスが間違っているか、exportが完了していない可能性があります。")
        return

    infer_request = compiled_model.create_infer_request()

    # クラス名リスト（見やすく整形しました）
    classes = {
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした。")
        return

    # カメラの解像度を720p (1280x720) に設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"カメラ解像度を {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} に設定しようと試みました。")
    
    print("=== Intel GPU リアルタイム検知 (OpenVINO Native) ===")
    print("終了するには 'q' キーを押してください。")

    # 人物が近づいたと判断するためのバウンディングボックスの高さのしきい値
    # この値は、カメラの解像度や設置場所によって調整が必要です
    # 720pの解像度で、1.5m程度まで近づいた人を検知することを想定 (画面高の約半分)
    PERSON_HEIGHT_THRESHOLD = 360

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 画像の前処理 (YOLOv8/12向け)
        # 1. 640x640にリサイズ（アスペクト比維持なしの簡易版）
        input_img = cv2.resize(frame, (640, 640))
        # 2. BGRからRGBへ変換
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # 3. [H, W, C] -> [C, H, W] へ変換し、バッチ次元を追加 [1, C, H, W]
        input_tensor = input_img.transpose(2, 0, 1)
        input_tensor = input_tensor[np.newaxis, :, :, :]
        # 4. 0-255 を 0.0-1.0 に正規化
        input_tensor = input_tensor.astype(np.float32) / 255.0

        # 推論実行 (GPU)
        results = infer_request.infer([input_tensor])

        # 出力取得 (通常は output0 という名前)
        detections = list(results.values())[0]
        # output shape: [1, 84, 8400] -> [4座標 + 80クラス]
        # これを [8400, 84] に転置して扱いやすくする
        detections = np.transpose(detections[0])

        boxes = []
        confidences = []
        class_ids = []

        # 画像サイズのスケーリング係数
        h_scale = frame.shape[0] / 640
        w_scale = frame.shape[1] / 640

        # 検出結果のフィルタリング
        # 信頼度が低いものは捨てる (閾値 0.5)
        scores = np.max(detections[:, 4:], axis=1)
        mask = scores > 0.5
        filtered_detections = detections[mask]

        for det in filtered_detections:
            class_id = np.argmax(det[4:])
            confidence = det[4 + class_id]

            # バウンディングボックス座標 (cx, cy, w, h)
            cx, cy, w, h = det[0], det[1], det[2], det[3]

            # 左上座標 (x1, y1) に変換し、元の画像サイズに戻す
            x1 = int((cx - w / 2) * w_scale)
            y1 = int((cy - h / 2) * h_scale)
            width = int(w * w_scale)
            height = int(h * h_scale)

            boxes.append([x1, y1, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

        # NMS (重複除去)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        # 描画
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = classes.get(class_ids[i], 'Unknown')
                conf = confidences[i]

                # ★★★ここから追加★★★
                # 検出したのが'person'で、かつ高さがしきい値を超えた場合
                if label == 'person' and h > PERSON_HEIGHT_THRESHOLD:
                    print("しきい値に達しました")
                # ★★★ここまで追加★★★

                # 枠とラベルを描画
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv12 - Intel GPU Native", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()