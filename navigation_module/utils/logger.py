import os
import logging
from logging.handlers import RotatingFileHandler

# 日志存放路径
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# 若路径不存在, 自动创建
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_logger(module_name: str) -> logging.Logger:
    """
    获取一个配置好的 Logger 实例
    :param module_name: 调用该日志的模块名称
    :return: logging.Logger 对象
    """
    # 创建日志实例
    logger = logging.getLogger(module_name)
    
    # 防止重复绑定
    if logger.handlers:
        return logger

    # 拦截等级 (DEBUG < INFO < WARNING < ERROR < CRITICAL)
    logger.setLevel(logging.DEBUG)

    # 输出格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # 文件处理器
    log_file_path = os.path.join(LOG_DIR, 'navigation_system.log')
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO) # 只存 INFO 及以上
    file_handler.setFormatter(formatter)

    # 绑定处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
