import base64
import mimetypes
import time
import cv2
from openai import OpenAI

client = OpenAI(
    api_key="sk-ydnrsfoiubdktfygaiuqstmpzhqjxqgqejvmkvizkwgzxunc",
    base_url="https://api.siliconflow.cn/v1"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_img_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    return mime_type

def process_image(data_url):
    stream = client.chat.completions.create(
        model="Pro/moonshotai/Kimi-K2.5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是实时视频流中的当前帧，请简要描述图片内容，主要描述当前图片相对于之前几帧的变化，如果图片内容与上一张图片相似，请直接回复“与上一张图片相似”。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ],
        stream=True,
        max_tokens=300,
    )

    for event in stream:
        print(event.choices[0].delta.content, end="", flush="True")
    print()

#image_path = "/home/firedust/test.png"
IMG_PROSCESS_FREQ = 1  

# base64_image = encode_image(image_path)
# mime_type = get_img_type(image_path)

# data_url = f"data:{mime_type};base64,{base64_image}"

cap = cv2.VideoCapture(0)

start_time = time.time()
current_time = start_time
last_time = start_time

while True:
    ret, frame = cap.read()
    if not ret:
        print('read video error')
        break

    cv2.imshow('frame', frame)

    current_time = time.time()
    if current_time - last_time > 1 / IMG_PROSCESS_FREQ :  
        last_time = current_time
        
        base64_image = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
        mime_type = "image/jpg"
        data_url = f"data:{mime_type};base64,{base64_image}"
        process_image(data_url)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
