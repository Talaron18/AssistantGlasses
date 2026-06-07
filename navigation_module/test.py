import sys
import os
import queue
import threading
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.nav_controller import NavController

def mock_tts_speaker(tts_queue):
    """模拟语音模块"""
    while True:
        text = tts_queue.get()
        if text == "SHUTDOWN_TTS":
            break
        print(f"\n=====================================")
        print(f"[语音播报触发] -> {text}")
        print(f"=====================================\n")
        tts_queue.task_done()

def main():
    print("\n正在启动导航模块独立测试...")
    
    # 建立通信队列
    nav_queue = queue.Queue()
    tts_queue = queue.Queue()

    # 语音输入模拟 (TTS线程)
    tts_thread = threading.Thread(target=mock_tts_speaker, args=(tts_queue,), daemon=True)
    tts_thread.start()

    # 导航核心控制器
    try:
        nav_core = NavController(nav_queue=nav_queue, tts_queue=tts_queue)
        nav_core.start()
    except Exception as e:
        print(f"导航模块启动失败，请检查硬件是否被占用: {e}")
        return

    print("导航线程已在后台运转")
    time.sleep(2)

    # === 原来只是输入 command，将其修改为扩充测试版 ===
    while True:
        print("\n模拟操作面板:")
        print("  [1] 导航: go 餐厅 (如: go 天安门)")
        print("  [2] 移动: loc 116.3301 39.9952 (手动把自己传送到此坐标)")
        print("  [3] 终止: STOP (模拟盲人喊出'结束导航')")
        print("  [4] 退出: EXIT")
        
        command = input(">>> ").strip()
        if not command:
            continue

        if command.upper() == "EXIT":
            print("正在安全关闭系统...")
            tts_queue.put("SHUTDOWN_TTS")
            nav_core.shutdown()
            break
        elif command.upper() == "STOP":
            nav_queue.put("STOP")
            
        elif command.lower().startswith("go "):
            # 把前缀干掉，抽取出地名丢给导航队列
            target_name = command[3:].strip()
            nav_queue.put(target_name)
            
        elif command.lower().startswith("loc "):
            # 处理 loc 坐标跃迁命令，例如 loc 116.397 39.908
            parts = command.split()
            if len(parts) == 3:
                try:
                    sim_lon = float(parts[1])
                    sim_lat = float(parts[2])
                    # 直接调用刚才留在 nav_controller 的后门
                    nav_core.force_set_position(sim_lon, sim_lat)
                    print(f"🌍 系统已强行把你带到了坐标: ({sim_lon}, {sim_lat})")
                except ValueError:
                    print("坐标格式错误，必须是两个浮点数字哦。")
            else:
                print("请输入两个坐标，例如: loc 116.330 39.990")
        else:
            print("不识别的命令。")

if __name__ == "__main__":
    main()