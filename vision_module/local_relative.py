from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import time
from pathlib import Path
import sys
import math

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    repo_root = Path(__file__).resolve().parent / "Depth-Anything-V2"
    if not repo_root.exists():
        raise ImportError(
            "找不到 depth_anything_v2 模块。\n"
            "请先执行 `uv add depth-anything-v2` 安装，"
            "或将官方仓库 Depth-Anything-V2 克隆到项目根目录。"
        )
    sys.path.append(str(repo_root))
    from depth_anything_v2.dpt import DepthAnythingV2


YOLO_DEVICE = "cpu"
YOLO_IMGSZ = 640

DEPTH_DEVICE = "cpu"
DEPTH_INPUT_SIZE = 384
DEPTH_INTERVAL_SEC = 0.5

yolo_model = YOLO("AssistantGlasses/checkpoints/yolo26s.pt", task="detect")
yolo_model.to(YOLO_DEVICE)

depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
state = torch.load("AssistantGlasses/checkpoints/depth_anything_v2_vits.pth", map_location=DEPTH_DEVICE)
depth_model.load_state_dict(state)
depth_model.to(DEPTH_DEVICE).eval()

CLASS_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
]


def main() -> None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("cannot open camera")
        return

    last_depth = None
    last_depth_vis = None
    last_depth_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("read video error")
            break

        now = time.time()

        if now - last_depth_time >= DEPTH_INTERVAL_SEC:
            with torch.no_grad():
                depth = depth_model.infer_image(frame, input_size=DEPTH_INPUT_SIZE)
            last_depth_time = now
            last_depth = depth

            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth_norm = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            last_depth_vis = cv.applyColorMap(depth_uint8, cv.COLORMAP_INFERNO)

        results = yolo_model.track(frame, stream=True, conf=0.4, imgsz=YOLO_IMGSZ, device=YOLO_DEVICE)

        yolo_vis = frame.copy()

        for result in results:
            boxes = result.boxes
            names = result.names

            per_class_counter = {}

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                conf = float(box.conf[0])

                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                if box.id is not None:
                    obj_id = int(box.id[0])
                else:
                    obj_id = per_class_counter.get(cls_id, 0)
                    per_class_counter[cls_id] = obj_id + 1

                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

                cv.rectangle(yolo_vis, (x1, y1), (x2, y2), color, 2)

                info = ""
                if last_depth is not None:
                    roi = last_depth[int(y1/3):int(y2/3), int(x1/3):int(x2/3)]
                    depth_val = float(np.median(roi))
                    info = f"d={depth_val:.2f}"

                label = f"{info}"
                text_y = max(y1 - 10, 0)
                cv.putText(
                    yolo_vis,
                    label,
                    (x1, text_y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv.LINE_AA,
                )

        cv.imshow("YOLO (objects)", yolo_vis)
        if last_depth_vis is not None:
            cv.imshow("Depth (relative distance)", last_depth_vis)

        time.sleep(0.01)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()