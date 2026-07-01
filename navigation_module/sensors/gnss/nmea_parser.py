import pynmea2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import get_logger

logger = get_logger(__name__)

class NMEAParser:
    def __init__(self):
        pass

    def parse(self, raw_line: str) -> dict:
        if not raw_line or not raw_line.startswith('$'):
            return None

        try:
            msg = pynmea2.parse(raw_line)

            if isinstance(msg, pynmea2.types.talker.RMC):
                is_valid = (msg.status == 'A')
                
                return {
                    "type": "RMC",
                    "is_valid": is_valid,
                    "latitude": msg.latitude,    
                    "longitude": msg.longitude,
                    "speed_kmh": float(msg.spd_over_grnd) * 1.852 if msg.spd_over_grnd else 0.0,
                    "true_course": float(msg.true_course) if msg.true_course else 0.0
                }

            elif isinstance(msg, pynmea2.types.talker.GGA):
                return {
                    "type": "GGA",
                    "satellites": int(msg.num_sats) if msg.num_sats else 0,
                    "hdop": float(msg.horizontal_dil) if msg.horizontal_dil else 99.9,
                    "altitude": float(msg.altitude) if msg.altitude else 0.0
                }

            return None

        except pynmea2.ParseError as e:
            logger.debug(f"丢弃残缺/乱码数据: {e}")
            return None
