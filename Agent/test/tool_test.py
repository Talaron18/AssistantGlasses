
import cv2
import os
from datetime import datetime


def quicktest(mode="ON"):
    """
    Camera tool for AssistantGlasses.

    Returns:
        {
            "status": "success",
            "image_path": "...jpg"
        }
    """

    if mode != "ON":
        return {
            "status": "disabled"
        }

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return {
            "status": "error",
            "message": "camera not found"
        }
    for _ in range(5):
        ret, frame = camera.read()
    ret, frame = camera.read()

    camera.release()

    if not ret:
        return {
            "status": "error",
            "message": "capture failed"
        }

    os.makedirs("captures", exist_ok=True)

    filename = datetime.now().strftime(
        "captures/%Y%m%d_%H%M%S.jpg"
    )

    cv2.imwrite(filename, frame)

    return {
        "status": "success",
        "image_path": filename
    }