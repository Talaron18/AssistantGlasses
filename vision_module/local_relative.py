from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import time
from pathlib import Path
import sys
import math

# 优雅导入 DepthAnythingV2（与 test.py 保持一致）：
try:
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore[import]
except ImportError:
    repo_root = Path(__file__).resolve().parent / "Depth-Anything-V2"
    if not repo_root.exists():
        raise ImportError(
            "找不到 depth_anything_v2 模块。\n"
            "请先执行 `uv add depth-anything-v2` 安装，"
            "或将官方仓库 Depth-Anything-V2 克隆到项目根目录。"
        )
    sys.path.append(str(repo_root))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore[import]


# ---------- 配置 ----------
# 两个模型都在 CPU 上跑，避免 GPU 依赖
YOLO_DEVICE = "cpu"
YOLO_IMGSZ = 640  # YOLO 输入分辨率，可酌情调小提速

DEPTH_DEVICE = "cpu"
DEPTH_INPUT_SIZE = 384
DEPTH_INTERVAL_SEC = 0.5  # 深度每 0.5s 计算一次（约 2 FPS）

# YOLO 模型
yolo_model = YOLO("AssistantGlasses/checkpoints/yolo26s.pt", task="detect")
# Ultralytics 内部会根据 device 参数切换设备，这里显式设为 CPU
yolo_model.to(YOLO_DEVICE)

# 深度模型
depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
state = torch.load("AssistantGlasses/checkpoints/depth_anything_v2_vits.pth", map_location=DEPTH_DEVICE)
depth_model.load_state_dict(state)
depth_model.to(DEPTH_DEVICE).eval()

# 预定义一些颜色，不同类别用不同颜色
CLASS_COLORS = [
    (0, 0, 255),      # 红
    (0, 255, 0),      # 绿
    (255, 0, 0),      # 蓝
    (0, 255, 255),    # 黄
    (255, 0, 255),    # 品红
    (255, 255, 0),    # 青
    (128, 0, 255),    # 紫
    (0, 128, 255),    # 橙
]


# def depth_to_distance(x: float) -> float:
#     """
#     将 Depth-Anything 的深度值 x 映射为距离 d。
#     分段函数：
#       - 0 < x < 0.64:   d = 20 / sqrt(x) + 292
#       - 0.64 <= x <= 7.65: d = 1.48725025*x^3 - 17.59782451*x^2 + 11.32317764*x + 316.5795694
#       - x > 7.65:       d = 80 * exp((-x + 5.5) / 3)
#     """
#     if x <= 0:
#         x = 1e-6  # 避免 sqrt(0) 或负数

#     if x < 0.64:
#         return 20.0 / math.sqrt(x) + 292.0
#     if x <= 7.65:
#         return (
#             1.48725025 * x**3
#             - 17.59782451 * x**2
#             + 11.32317764 * x
#             + 316.5795694
#         )
#     return 80.0 * math.exp((-x + 5.5) / 3.0)


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

        # ---------- 深度推理（低频） ----------
        if now - last_depth_time >= DEPTH_INTERVAL_SEC:
            with torch.no_grad():
                depth = depth_model.infer_image(frame, input_size=DEPTH_INPUT_SIZE)
            last_depth_time = now
            last_depth = depth

            # 为 depth 打伪彩色，单独一个窗口显示
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth_norm = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            last_depth_vis = cv.applyColorMap(depth_uint8, cv.COLORMAP_INFERNO)

        # ---------- YOLO 检测（高频） ----------
        # 为了控制 CPU 时间，这里可以按需降低帧率：例如只处理每隔 N 帧
        results = yolo_model.track(frame, stream=True, conf=0.4, imgsz=YOLO_IMGSZ, device=YOLO_DEVICE)

        yolo_vis = frame.copy()

        for result in results:
            boxes = result.boxes
            names = result.names  # 模型内置的类别名字典

            # 每帧内，为同一类别的目标分配一个递增 id（从 0 开始）
            per_class_counter = {}

            for box in boxes:
                # 目标坐标
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                # 置信度
                conf = float(box.conf[0])

                # 类别 id 与类别名称
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                # 使用模型自带的 track id（如果有），否则按同类别顺序编号
                if box.id is not None:
                    obj_id = int(box.id[0])
                else:
                    obj_id = per_class_counter.get(cls_id, 0)
                    per_class_counter[cls_id] = obj_id + 1

                # 不同类别用不同颜色
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

                # 画框（画在 yolo_vis 上）
                cv.rectangle(yolo_vis, (x1, y1), (x2, y2), color, 2)

                # 如果有最新深度图，取框中心的深度值，并换算为距离 d
                info = ""
                if last_depth is not None:
                    roi = last_depth[int(y1/3):int(y2/3), int(x1/3):int(x2/3)]
                    depth_val = float(np.median(roi))
                    # dist = depth_to_distance(depth_val)
                    info = f"d={depth_val:.2f}"

                # 显示：类别名、同类别中的 id、置信度 + 换算后距离 d
                #label = f"{class_name}-{obj_id} {conf:.2f}{info}"
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

        # 两个窗口分别展示 YOLO 和 Depth 结果
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