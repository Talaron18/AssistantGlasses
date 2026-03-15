import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sensors.base_sensor import BaseSensor
from utils.logger import get_logger

logger = get_logger(__name__)

class MockGNSSReader(BaseSensor):
    """
    数据回放/模拟读取器
    在没有物理 GPS 硬件时，它从指定的 txt 文件中逐行读取 NMEA 数据
    模拟真实的硬件输出，方便软件逻辑的开发和调试
    """
    def __init__(self, mock_file_name="nmea_sample.txt"):
        self.is_connected = False
        self.lines =[]
        self.current_index = 0
        
        # 定位测试数据文件绝对路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.file_path = os.path.join(project_root, 'tests', 'mock_data', mock_file_name)
        
        self._connect()

    def _connect(self):
        """模拟硬件连接"""
        if not os.path.exists(self.file_path):
            logger.error(f"找不到 Mock 数据文件: {self.file_path}")
            return
            
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
            self.is_connected = True
            logger.info(f"[Mock模式] 成功加载回放数据: {os.path.basename(self.file_path)}，共 {len(self.lines)} 行")
        except Exception as e:
            logger.error(f"读取 Mock 数据失败: {e}")

    def read_data(self) -> str:
        """
        模拟串口按行读取数据
        模拟硬件的波特率延迟
        """
        if not self.is_connected:
            return None
            
        if self.current_index >= len(self.lines):
            # 循环播放
            logger.warning(" [Mock模式] 文件已读完，重新循环播放数据...")
            self.current_index = 0
            time.sleep(1) #
            
        line = self.lines[self.current_index].strip()
        self.current_index += 1
        
        # 模拟物理串口传输延迟
        time.sleep(0.1) 
        
        return line

    def health_check(self) -> bool:
        """文件还在内存中，模拟设备在线"""
        return self.is_connected

    def close(self):
        """模拟释放资源"""
        self.is_connected = False
        self.lines.clear()
        logger.info("[Mock模式] 虚拟串口已关闭")