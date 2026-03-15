import serial
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sensors.base_sensor import BaseSensor
from utils.logger import get_logger
from config.config_loader import load_config

logger = get_logger(__name__)

class GNSSSerialReader(BaseSensor):
    def __init__(self):
        # 加载配置
        config = load_config()
        self.port = config['hardware']['gnss']['port']
        self.baud_rate = config['hardware']['gnss']['baud_rate']
        self.timeout = config['hardware']['gnss']['timeout']
        
        self.serial_conn = None
        self.is_connected = False
        
        # 非阻塞读取缓冲区
        self._buffer = "" 
        
        # 打开物理串口
        self._connect()

    def _connect(self):
        """内部方法：执行物理串口连接"""
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

    def read_data(self) -> str:
        """
        非阻塞式按行读取串口数据
        :return: 一行完整的 NMEA 字符串, 若无完整数据或失败返回 None
        """
        if not self.is_connected or self.serial_conn is None:
            return None

        try:
            # 1. 检查操作系统底层缓冲区是否有数据准备好
            if self.serial_conn.in_waiting > 0:
                # 读出所有可用字节，不阻塞
                raw_data = self.serial_conn.read(self.serial_conn.in_waiting)
                # 拼接到内部缓冲区
                self._buffer += raw_data.decode('ascii', errors='ignore')

            # 检查缓冲区内是否拼接成了一个完整的行
            if '\n' in self._buffer:
                # 分割出第一行，剩下的留给下一次
                line, self._buffer = self._buffer.split('\n', 1)
                clean_line = line.strip()
                if clean_line:
                    return clean_line

            # 若没有完整的一行，瞬间返回 None
            return None
            
        except serial.SerialException as e:
            logger.error(f"读取数据时串口断开: {e}")
            self.is_connected = False
            return None
        except Exception as e:
            logger.error(f"串口读取发生未知异常: {e}")
            return None

    def health_check(self) -> bool:
        """检查串口是否依然处于打开状态"""
        return self.is_connected and self.serial_conn.is_open

    def close(self):
        """安全关闭串口，释放计算机资源"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            self._buffer = ""
            logger.info("串口已安全关闭")
