import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nav_controller import NavController
from utils.logger import get_logger

logger = get_logger("Main")

def main():
    logger.info("导航模块启动中...")

    controller = NavController()

    try:
        controller.run_main_loop()
        
    except KeyboardInterrupt:
        logger.warning("手动退出")
    except Exception as e:
        logger.error(f"系统发生未捕获的崩溃异常: {e}")
    finally:
        controller.shutdown()  
if __name__ == "__main__":
    main()