import yaml
import os
import sys
from functools import lru_cache
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)

PLACEHOLDER_KEYS = frozenset({"", "AMAP_KEY", "API_KEY", "YOUR_AMAP_KEY_HERE"})


def is_valid_api_key(key) -> bool:
    """判断 API Key 是否为可用的真实密钥 (非空且不属于占位符)。"""
    return isinstance(key, str) and key.strip() not in PLACEHOLDER_KEYS


@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    加载并解析 system_config.yaml 文件，并注入环境变量以保证安全。
    """
    load_dotenv()
    config_path = os.path.join(os.path.dirname(__file__), 'system_config.yaml')

    if not os.path.exists(config_path):
        logger.error(f"未找到配置文件 {config_path}")
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)

            env_amap_key = os.environ.get('AMAP_API_KEY')
            if env_amap_key:
                config_dict['api']['amap']['key'] = env_amap_key
                logger.info("已安全从环境变量加载高德地图 API Key")
            else:
                current_key = config_dict.get('api', {}).get('amap', {}).get('key', '')
                if not is_valid_api_key(current_key):
                    logger.warning(
                        "未检测到有效的 API Key! 请在终端设置 AMAP_API_KEY 环境变量"
                    )
                else:
                    logger.warning(
                        "正在使用 YAML 中的明文 API Key, 建议改用环境变量配置。"
                    )

            logger.info("系统配置文件加载成功")
            return config_dict

        except yaml.YAMLError as e:
            logger.error(f"YAML 文件解析错误: {e}")
            raise