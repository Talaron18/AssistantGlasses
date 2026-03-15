
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
import time
from pathlib import Path
import sys
import matplotlib

# 优雅导入 DepthAnythingV2（与 test.py 保持一致）：
try:
    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2  # type: ignore[import]
except ImportError:
    repo_root = Path(__file__).resolve().parent / "Depth-Anything-V2"
    if not repo_root.exists():
        raise ImportError(
            "找不到 depth_anything_v2 模块。\n"
            "请先执行 `uv add depth-anything-v2` 安装，"
            "或将官方仓库 Depth-Anything-V2 克隆到项目根目录。"
        )
    sys.path.append(str(repo_root))
    from metric_depth.depth_anything_v2.dpt import DepthAnythingV2  # type: ignore[import]


# ---------- 配置 ----------
# 两个模型都在 CPU 上跑，避免 GPU 依赖
YOLO_DEVICE = "cpu"
YOLO_IMGSZ = 640  # YOLO 输入分辨率，可酌情调小提速

DEPTH_DEVICE = "cpu"
DEPTH_INPUT_SIZE = 384
DEPTH_INTERVAL_SEC = 5  # 深度每 5s 计算一次（约 0.2 FPS）

# YOLO 模型
yolo_model = YOLO("AssistantGlasses/checkpoints/yolo26s.pt", task="detect")
# Ultralytics 内部会根据 device 参数切换设备，这里显式设为 CPU
yolo_model.to(YOLO_DEVICE)

# 深度模型 (metric depth, 参考 Depth-Anything-V2/metric_depth/run.py)
DEPTH_ENCODER = "vitb" # 可选 "vits", "vitb", "vitl", "vitg"，根据需要选择不同大小的模型，注意要对应加载正确的权重文件
DEPTH_MAX = 20.0
DEPTH_CHECKPOINT = f"AssistantGlasses/checkpoints/depth_anything_v2_metric_vkitti_{DEPTH_ENCODER}.pth"

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

depth_model = DepthAnythingV2(**MODEL_CONFIGS[DEPTH_ENCODER], max_depth=DEPTH_MAX)
state = torch.load(DEPTH_CHECKPOINT, map_location=DEPTH_DEVICE)
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
NUM_CLASSES = 2
NUM_REGIONS = 3
TARGET_CLASSES = {"person": 0, "car": 1}  # 只关注人和车，类别索引

def detect_region_objects(
        boxes,
        names,
        depth_map,
        region_polys,
        threshold_dist,
        frame_shape
):
    h, w = frame_shape[:2]

    region_has = [[0]*NUM_CLASSES for _ in range(NUM_REGIONS)]
    region_min_dist = [[float('inf')]*NUM_CLASSES for _ in range(NUM_REGIONS)]

    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls_id = int(box.cls[0])

        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        if class_name not in TARGET_CLASSES:
            continue

        class_idx = TARGET_CLASSES[class_name]

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        depth_val = float('nan')

        if depth_map is not None:
            cx = np.clip(cx,0,w-1)
            cy = np.clip(cy,0,h-1)
            depth_val = float(depth_map[cy,cx])

        for r, poly in enumerate(region_polys):
            poly = np.array(poly, dtype=np.float32).reshape((-1, 2))
            pt = (float(cx), float(cy))
            if cv.pointPolygonTest(poly, pt, False) >= 0:
                region_has[r][class_idx] = 1

                if not np.isnan(depth_val):
                    region_min_dist[r][class_idx] = min(
                        region_min_dist[r][class_idx],
                        depth_val
                    )
                break

    region_too_close = [
        [
            1 if region_has[r][c] and region_min_dist[r][c] < threshold_dist else 0
            for c in range(NUM_CLASSES)
        ]
        for r in range(NUM_REGIONS)
    ]

    return region_has, region_min_dist, region_too_close

def main() -> None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("cannot open camera")
        return

    last_depth = None
    depth_vis = None
    last_depth_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("read video error")
            break

        now = time.time()

        cmap = matplotlib.colormaps.get_cmap('Spectral')

        # ---------- 深度推理（低频） ----------
        if now - last_depth_time >= DEPTH_INTERVAL_SEC:
            with torch.no_grad():
                depth = depth_model.infer_image(frame, input_size=DEPTH_INPUT_SIZE)
            last_depth_time = now
            last_depth = depth

            # 为 depth 打伪彩色，单独一个窗口显示
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_uint8 = depth.astype(np.uint8)
            
            depth_vis = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # ---------- YOLO 检测（高频） ----------
        # 为了控制 CPU 时间，这里可以按需降低帧率：例如只处理每隔 N 帧
        results = yolo_model.track(frame, stream=True, conf=0.4, imgsz=YOLO_IMGSZ, device=YOLO_DEVICE)

        yolo_vis = frame.copy()

        # 区域划分（底边三等分 + 上角连线）
        h, w = frame.shape[:2]
        tl = (0, 0)
        tr = (w - 1, 0)
        bl = (0, h - 1)
        br = (w - 1, h - 1)
        b1 = (w // 3, h - 1)
        b2 = (2 * w // 3, h - 1)

        # 3 个区域多边形定义
        region_polys = [
            np.array([bl, b1, tl], dtype=np.int32),
            np.array([b1, b2, tr, tl], dtype=np.int32),
            np.array([b2, br, tr], dtype=np.int32),
        ]

        # 画区域边缘
        for poly in region_polys:
            cv.polylines(yolo_vis, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

        # 将深度图缩放到原图大小便于坐标映射
        depth_for_vis = None
        if last_depth is not None:
            if last_depth.shape[:2] != (h, w):
                depth_for_vis = cv.resize(last_depth, (w, h), interpolation=cv.INTER_LINEAR)
            else:
                depth_for_vis = last_depth

        # 存储每个区域是否有人以及最小距离
        threshold_dist = 3  # 米，危险阈值

        region_too_close = [[0] * NUM_CLASSES for _ in range(NUM_REGIONS)]

        for result in results:
            boxes = result.boxes
            names = result.names  # 模型内置的类别名字典

            h, w = frame.shape[:2]

            region_has = [[0]*NUM_CLASSES for _ in range(NUM_REGIONS)]
            region_min_dist = [[float('inf')]*NUM_CLASSES for _ in range(NUM_REGIONS)]

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls_id = int(box.cls[0])

                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                if class_name not in TARGET_CLASSES:
                    continue

                class_idx = TARGET_CLASSES[class_name]

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                depth_val = float('nan')

                if depth_for_vis is not None:
                    cx = np.clip(cx,0,w-1)
                    cy = np.clip(cy,0,h-1)
                    depth_val = float(depth_for_vis[cy,cx])

                #画框和人+距离信息 
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)] 
                cv.rectangle(yolo_vis, (x1, y1), (x2, y2), color, 2) 
                info = f"{class_name}, d={depth_val:.2f}" if not np.isnan(depth_val) else class_name 
                text_y = max(y1 - 10, 0) 
                cv.putText( 
                    yolo_vis, 
                    info, 
                    (x1, text_y), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color, 
                    2, 
                    cv.LINE_AA, 
                    )
                
                for r, poly in enumerate(region_polys):
                    poly = np.array(poly, dtype=np.float32).reshape((-1, 2))
                    pt = (float(cx), float(cy))
                    if cv.pointPolygonTest(poly, pt, False) >= 0:
                        region_has[r][class_idx] = 1

                        if not np.isnan(depth_val):
                            region_min_dist[r][class_idx] = min(
                                region_min_dist[r][class_idx],
                                depth_val
                            )
                        break


            region_too_close = [
                [
                    1 if region_has[r][c] and region_min_dist[r][c] < threshold_dist else 0
                    for c in range(NUM_CLASSES)
                ]
                for r in range(NUM_REGIONS)
            ]


        left_person, left_vehicle = region_too_close[0]
        mid_person, mid_vehicle = region_too_close[1]
        right_person, right_vehicle = region_too_close[2]

        advice = "请向前走"

        # 中间危险
        if mid_person or mid_vehicle:

            if left_person or left_vehicle and right_person or right_vehicle:
                advice = "前方危险，请停止或后退"

            elif left_person or left_vehicle:
                advice = "前方障碍，请向右前方绕行"

            elif right_person or right_vehicle:
                advice = "前方障碍，请向左前方绕行"

            else:
                advice = "前方有障碍，请绕行"

        # 左右都有
        elif (left_person or left_vehicle) and (right_person or right_vehicle):

            advice = "左右前方均有障碍，请减速"

        elif left_person or left_vehicle:

            advice = "左前方有障碍，请向右前方行驶"

        elif right_person or right_vehicle:

            advice = "右前方有障碍，请向左前方行驶"

        else:

            advice = "前方安全，请继续前行"

        print(advice)
        # 两个窗口分别展示 YOLO 和 Depth 结果
        cv.imshow("YOLO (objects)", yolo_vis)
        if depth_vis is not None:
            cv.imshow("Depth (absolute distance)", depth_vis)

        time.sleep(0.01)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()