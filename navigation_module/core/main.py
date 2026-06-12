import sys
import os
import queue
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nav_controller import NavController
from utils.logger import get_logger

logger = get_logger("Main")


def main(tts_queue: queue.Queue, nav_queue: queue.Queue) -> None:
    """
    启动导航后台线程。
    """
    logger.info("导航模块启动中...")

    controller = NavController(nav_queue=nav_queue, tts_queue=tts_queue)
    controller.start()  # 后台 daemon 线程, 执行 NavController.run()

    try:
        # 保持主线程存活; join 带超时使 Ctrl+C 能被及时捕获
        while controller.is_alive():
            controller.join(timeout=1.0)

    except KeyboardInterrupt:
        logger.warning("手动退出")
    except Exception as e:
        logger.error(f"系统发生未捕获的崩溃异常: {e}")
    finally:
        controller.shutdown()


def _debug_tts_consumer(tts_queue: queue.Queue) -> None:
    """独立运行时的简易 TTS 消费者: 把语音内容打印到终端, 防止队列无限堆积。"""
    while True:
        msg = tts_queue.get()
        logger.info(f"[TTS 播报] {msg}")


if __name__ == "__main__":
    # 独立运行时自行构造两个队列, 并启动一个调试用 TTS 打印线程。
    _tts_q: queue.Queue = queue.Queue()
    _nav_q: queue.Queue = queue.Queue()

    threading.Thread(target=_debug_tts_consumer, args=(_tts_q,), daemon=True).start()

    main(tts_queue=_tts_q, nav_queue=_nav_q)