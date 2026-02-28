from openai import OpenAI

def main():
    client = OpenAI(
        api_key="sk-ydnrsfoiubdktfygaiuqstmpzhqjxqgqejvmkvizkwgzxunc",
        base_url="https://api.siliconflow.cn/v1"
    )
    
    stream = client.chat.completions.create(
        model="Pro/zai-org/GLM-4.7",
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "你的上下文管理策略是什么？还是说你每次只处理当前输入，不考虑之前的对话？"}
        ],
        stream=True
    )
    
    for event in stream:
        print(event.choices[0].delta.content, end="", flush=True)
    print()
    
if __name__ == "__main__":
    main()
# import cv2 as cv

# cap = cv.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print('read video error')
#         break

#     cv.imshow('frame', frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()