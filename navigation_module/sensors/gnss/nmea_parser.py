import pynmea2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import get_logger

logger = get_logger(__name__)

class NMEAParser:
    """
    NMEA-0183 协议解析器
    将字符串转换为字典数据
    """
    def __init__(self):
        pass

    def parse(self, raw_line: str) -> dict:
        """
        解析单行 NMEA 数据
        :param raw_line: 示例 "$GPRMC,083559.00,A,4717.11437,N..."
        :return: 包含关键信息的字典, 无效数据返回 None
        """
        if not raw_line or not raw_line.startswith('$'):
            return None

        try:
            # 使用 pynmea2 解析
            msg = pynmea2.parse(raw_line)

            # GPRMC (经纬度、速度、航向、数据有效性)
            if isinstance(msg, pynmea2.types.talker.RMC):
                # 'A' 有效，'V' 无效
                is_valid = (msg.status == 'A')
                
                return {
                    "type": "RMC",
                    "is_valid": is_valid,
                    "latitude": msg.latitude,    
                    "longitude": msg.longitude,
                    "speed_kmh": float(msg.spd_over_grnd) * 1.852 if msg.spd_over_grnd else 0.0,
                    "true_course": float(msg.true_course) if msg.true_course else 0.0
                }

            # GPGGA (卫星数量、HDOP、海拔) 
            elif isinstance(msg, pynmea2.types.talker.GGA):
                return {
                    "type": "GGA",
                    "satellites": int(msg.num_sats) if msg.num_sats else 0,
                    "hdop": float(msg.horizontal_dil) if msg.horizontal_dil else 99.9,
                    "altitude": float(msg.altitude) if msg.altitude else 0.0
                }

            return None

        except pynmea2.ParseError as e:
            # 丢弃意外脏数据
            logger.debug(f"丢弃残缺/乱码数据: {e}")
            return None
