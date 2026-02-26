import yaml
import os
import sys

# 将工程根目录加入系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config() -> dict:
    """
    加载并解析 system_config.yaml 文件
    :return: 包含所有配置项的字典
    """
    # 动态获取 yaml 文件的绝对路径
    config_path = os.path.join(os.path.dirname(__file__), 'system_config.yaml')
    
    if not os.path.exists(config_path):
        logger.error(f"未找到配置文件 {config_path}")
        raise FileNotFoundError(f"Missing config file: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
            logger.info("系统配置文件加载成功")
            return config_dict
        except yaml.YAMLError as e:
            logger.error(f"YAML 文件解析错误: {e}")
            raise
