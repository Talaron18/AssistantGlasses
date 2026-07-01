import serial
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sensors.base_sensor import BaseSensor
from utils.logger import get_logger
from config.config_loader import load_config

logger = get_logger(__name__)

RECONNECT_INTERVAL_S = 5.0


class GNSSSerialReader(BaseSensor):
    def __init__(self):
        config = load_config()
        self.port = config['hardware']['gnss']['port']
        self.baud_rate = config['hardware']['gnss']['baud_rate']
        self.timeout = config['hardware']['gnss']['timeout']

        self.serial_conn = None
        self.is_connected = False

        self._buffer = ""

        self._last_reconnect_ts = 0.0

        self._connect()

    def _connect(self):
        self._last_reconnect_ts = time.monotonic()
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.is_connected = True
            self._buffer = ""
            logger.info(f"成功连接至 GNSS 模块: {self.port} @ {self.baud_rate}bps")
        except serial.SerialException as e:
            self.is_connected = False
            logger.error(f"无法打开串口 {self.port}")
            logger.debug(f"错误信息: {e}")

    def _try_reconnect(self):
        if time.monotonic() - self._last_reconnect_ts < RECONNECT_INTERVAL_S:
            return
        logger.info(f"尝试重新连接 GNSS 模块 {self.port} …")
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        self._connect()

    def read_data(self) -> str:
        if not self.is_connected or self.serial_conn is None:
            self._try_reconnect()
            if not self.is_connected:
                return None

        try:
            if self.serial_conn.in_waiting > 0:
                raw_data = self.serial_conn.read(self.serial_conn.in_waiting)
                self._buffer += raw_data.decode('ascii', errors='ignore')

            if '\n' in self._buffer:
                line, self._buffer = self._buffer.split('\n', 1)
                clean_line = line.strip()
                if clean_line:
                    return clean_line

            return None

        except serial.SerialException as e:
            logger.error(f"读取数据时串口断开: {e}")
            self.is_connected = False
            return None
        except Exception as e:
            logger.error(f"串口读取发生未知异常: {e}")
            return None

    def health_check(self) -> bool:
        return self.is_connected and self.serial_conn is not None and self.serial_conn.is_open

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            self._buffer = ""
            logger.info("串口已安全关闭")