import base64
from io import BytesIO
import cv2

# pillow-image to base64
def pil_to_base64(img):
    buffered=BytesIO()
    img.save(buffered,format="JPG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# opencv-image to base64
def cv_to_base64(img):
    checkpoint,buffered=cv2.imencode(".jpg",img)
    if not checkpoint:
        return None
    return base64.b64encode(buffered).decode("utf-8")

# stored image to base64
def img_to_base64(img_path):
    with open(img_path) as img:
        base64_string=base64.b64encode(img.read())
        return base64_string.decode("utf-8")

def to_base64(img,input_type="cv"):
    if input_type=="cv":
        processed_img=cv_to_base64(img)
    elif input_type=="pil":
        processed_img=pil_to_base64(img)
    return processed_img