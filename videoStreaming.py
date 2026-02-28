import base64                                                                                                                                                                           
import time                                                                                                                                                                             
import cv2                                                                                                                                                                              
import threading                                                                                                                                                                        
from openai import OpenAI                                                                                                                                                               
from collections import deque                                                                                                                                                           
                                                                                                                                                                                           
client = OpenAI(                                                                                                                                                                        
       api_key="sk-ydnrsfoiubdktfygaiuqstmpzhqjxqgqejvmkvizkwgzxunc",                                                                                                                     
       base_url="https://api.siliconflow.cn/v1"                                                                                                                                            
   )                                                                                                                                                                                       
                                                                                                                                                                                           
IMG_PROCESS_FREQ = 1  # 每秒处理次数                                                                                                                                                    
MAX_HISTORY_TURNS = 10  # 保留最近10轮文本对话                                                                                                                                          
                                                                                                                                                                                           
   # ===== 共享变量 =====                                                                                                                                                                  
shared_frame = None                                                                                                                                                                     
frame_lock = threading.Lock()                                                                                                                                                           
running = True                                                                                                                                                                          
                                                                                                                                                                                           
   # ===== 对话记忆（仅文本，不含图片）=====                                                                                                                                               
   # 使用 deque 自动限制长度，无需手动裁剪                                                                                                                                                 
text_history = deque(maxlen=MAX_HISTORY_TURNS * 2)  # 每轮2条：user+assistant                                                                                                           
                                                                                                                                                                                           
SYSTEM_PROMPT = """你是一个实时盲人导航辅助系统。                                                                                                                                       
                                                                                                                                                                                           
   目标：                                                                                                                                                                                  
   为盲人提供实时避障语音指令。                                                                                                                                                            
                                                                                                                                                                                           
   感知优先级：                                                                                                                                                                            
   1. 正前方1-3米内障碍物（最高优先级）                                                                                                                                                    
   2. 动态物体（行人、车辆、自行车）                                                                                                                                                       
   3. 地面变化（台阶、坑洞、积水）                                                                                                                                                         
   4. 窄路、路边障碍                                                                                                                                                                       
                                                                                                                                                                                           
   忽略：                                                                                                                                                                                  
   - 颜色                                                                                                                                                                                  
   - 风景                                                                                                                                                                                  
   - 建筑外观                                                                                                                                                                              
   - 广告牌                                                                                                                                                                                
                                                                                                                                                                                           
   输出规则：                                                                                                                                                                              
   - 不超过15字                                                                                                                                                                            
   - 直接给行动指令                                                                                                                                                                        
   - 不能解释原因                                                                                                                                                                          
   - 不能描述画面                                                                                                                                                                          
   - 若安全：前方安全"""                                                                                                                                                                   
                                                                                                                                                                                           
                                                                                                                                                                                           
def build_messages_with_image(data_url: str) -> list:                                                                                                                                  
    """                                                                                                                                                                                 
    构建包含当前图片的完整消息列表。                                                                                                                                                    
    历史对话只保留文本，图片仅用于当前帧。                                                                                                                                              
    """                                                                                                                                                                                 
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]                                                                                                                           
                                                                                                                                                                                           
       # 添加历史文本对话（不含图片）                                                                                                                                                      
    messages.extend(list(text_history))                                                                                                                                                
                                                                                                                                                                                           
       # 添加当前帧（带图片）                                                                                                                                                              
    messages.append({                                                                                                                                                                   
           "role": "user",                                                                                                                                                                 
           "content": [                                                                                                                                                                    
               {"type": "text", "text": "这是当前实时画面，请判断是否存在行走风险"},                                                                                                       
               {"type": "image_url", "image_url": {"url": data_url}}                                                                                                                       
           ]                                                                                                                                                                               
       })                                                                                                                                                                                  
                                                                                                                                                                                           
    return messages                                                                                                                                                                     
                                                                                                                                                                                           
                                                                                                                                                                                           
def process_image(data_url: str):                                                                                                                                                       
       """                                                                                                                                                                                 
       处理单帧图像，调用视觉API。                                                                                                                                                         
       """                                                                                                                                                                                 
       global text_history                                                                                                                                                                 
                                                                                                                                                                                           
       messages = build_messages_with_image(data_url)                                                                                                                                     
                                                                                                                                                                                           
       try:                                                                                                                                                                                
           stream = client.chat.completions.create(                                                                                                                                        
               model="Qwen/Qwen3-VL-8B-Instruct",                                                                                                                                          
               messages=messages,                                                                                                                                                          
               stream=True,                                                                                                                                                                
               max_tokens=150,                                                                                                                                                             
           )                                                                                                                                                                               
                                                                                                                                                                                           
           print("\nAI:", end=" ", flush=True)                                                                                                                                             
                                                                                                                                                                                           
           response_text = ""                                                                                                                                                              
           for event in stream:                                                                                                                                                            
               if event.choices[0].delta.content:                                                                                                                                          
                   chunk = event.choices[0].delta.content                                                                                                                                  
                   print(chunk, end="", flush=True)                                                                                                                                        
                   response_text += chunk                                                                                                                                                  
                                                                                                                                                                                           
           print()                                                                                                                                                                         
                                                                                                                                                                                           
           # 只保存文本到历史（关键：不存图片！）                                                                                                                                          
           text_history.append({                                                                                                                                                           
               "role": "user",                                                                                                                                                             
               "content": "这是当前实时画面，请判断是否存在行走风险"                                                                                                                       
           })                                                                                                                                                                              
           text_history.append({                                                                                                                                                           
               "role": "assistant",                                                                                                                                                        
               "content": response_text                                                                                                                                                    
           })                                                                                                                                                                              
                                                                                                                                                                                           
       except Exception as e:                                                                                                                                                              
           print("API error:", e)                                                                                                                                                          
                                                                                                                                                                                           
                                                                                                                                                                                           
   # ===== 线程1：摄像头采集 =====                                                                                                                                                         
def capture_thread():                                                                                                                                                                   
       global shared_frame, running                                                                                                                                                        
                                                                                                                                                                                           
       cap = cv2.VideoCapture(0)                                                                                                                                                           
                                                                                                                                                                                           
       if not cap.isOpened():                                                                                                                                                              
           print("Cannot open camera")                                                                                                                                                     
           running = False                                                                                                                                                                 
           return                                                                                                                                                                          
                                                                                                                                                                                           
       while running:                                                                                                                                                                      
           ret, frame = cap.read()                                                                                                                                                         
           if not ret:                                                                                                                                                                     
               print("Read video error")                                                                                                                                                   
               break                                                                                                                                                                       
                                                                                                                                                                                           
           with frame_lock:                                                                                                                                                                
               shared_frame = frame.copy()                                                                                                                                                 
                                                                                                                                                                                           
           cv2.imshow('frame', frame)                                                                                                                                                      
                                                                                                                                                                                           
           if cv2.waitKey(1) & 0xFF == ord('q'):                                                                                                                                           
               running = False                                                                                                                                                             
               break                                                                                                                                                                       
                                                                                                                                                                                           
       cap.release()                                                                                                                                                                       
       cv2.destroyAllWindows()                                                                                                                                                             
                                                                                                                                                                                           
                                                                                                                                                                                           
   # ===== 线程2：API处理 =====                                                                                                                                                            
def api_thread():                                                                                                                                                                       
       global shared_frame, running                                                                                                                                                        
                                                                                                                                                                                           
       last_time = time.time()                                                                                                                                                             
                                                                                                                                                                                           
       while running:                                                                                                                                                                      
           current_time = time.time()                                                                                                                                                      
                                                                                                                                                                                           
           if current_time - last_time > 1 / IMG_PROCESS_FREQ:                                                                                                                             
               last_time = current_time                                                                                                                                                    
                                                                                                                                                                                           
               with frame_lock:                                                                                                                                                            
                   if shared_frame is None:                                                                                                                                                
                       continue                                                                                                                                                            
                   frame_copy = shared_frame.copy()                                                                                                                                        
                                                                                                                                                                                           
               # 编码图片（设置质量减少大小）                                                                                                                                              
               _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 70])                                                                                                
               base64_image = base64.b64encode(buffer).decode( 'utf-8')                                                                                                                    
               data_url = f"data:image/jpeg;base64,{base64_image}"                                                                                                                         
                                                                                                                                                                                           
               process_image(data_url)                                                                                                                                                     
                                                                                                                                                                                           
           time.sleep(0.01)                                                                                                                                                                
                                                                                                                                                                                           
                                                                                                                                                                                           
   # ===== 主程序 =====                                                                                                                                                                    
if __name__ == "__main__":                                                                                                                                                              
       CapThread = threading.Thread(target=capture_thread, daemon=True)                                                                                                                   
       APIThread = threading.Thread(target=api_thread, daemon=True)                                                                                                                       
                                                                                                                                                                                           
       CapThread.start()                                                                                                                                                                   
       APIThread.start()                                                                                                                                                                   
                                                                                                                                                                                           
       try:                                                                                                                                                                                
           CapThread.join()                                                                                                                                                                
           APIThread.join()                                                                                                                                                                
       except KeyboardInterrupt:                                                                                                                                                           
           running = False                                                                                                                                                                 
           print("\nInterrupted.")                                                                                                                                                         
                                                                                                                                                                                           
       print("Program exited.") 