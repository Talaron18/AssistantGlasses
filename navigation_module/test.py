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

    # 模拟大模型和用户的交互循环
    while True:
        print("\n模拟操作:")
        print("  -> 输入目的地名称 (如: 良乡大学城北地铁站)")
        print("  -> 输入 STOP (模拟盲人喊出'结束导航')")
        print("  -> 输入 EXIT (退出本测试工具)")
        
        command = input(">>> ").strip()

        if command.upper() == "EXIT":
            print("正在安全关闭系统...")
            tts_queue.put("SHUTDOWN_TTS")
            nav_core.shutdown()
            break
        elif command:
            # 将你的输入塞进队列
            nav_queue.put(command)

if __name__ == "__main__":
    main()