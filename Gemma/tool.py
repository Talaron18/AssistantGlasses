import base64
import time
from datetime import datetime
from pathlib import Path

import cv2


_DEFAULT_PHOTO_DIR = Path("./photos")


def capture_photo(save_dir: str = str(_DEFAULT_PHOTO_DIR)) -> dict:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)  # 0 = OS default camera
    if not cap.isOpened():
        return {
            "success": False,
            "error": (
                "Cannot open the default camera (index 0). "
                "Make sure no other application is using it."
            ),
        }

    try:
        # Warm-up: discard a few frames so auto-exposure / white-balance settle
        for _ in range(5):
            cap.read()
        time.sleep(0.3)

        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        return {
            "success": False,
            "error": "Camera opened but failed to capture a frame.",
        }

    # ---- Save to disk -------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"photo_{timestamp}.jpg"
    filepath  = output_dir / filename

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        return {"success": False, "error": "Failed to encode frame as JPEG."}

    try:
        filepath.write_bytes(buf.tobytes())
    except OSError as exc:
        return {"success": False, "error": f"Could not write image to {filepath}: {exc}"}

    # ---- Base64 for inline vision input ------------------------------------
    b64 = base64.b64encode(buf).decode("utf-8")

    print(f"[capture_photo] Saved → {filepath.resolve()}")
    return {
        "success":  True,
        "path":     str(filepath.resolve()),
        "filename": filename,
        "base64":   b64,
    }
