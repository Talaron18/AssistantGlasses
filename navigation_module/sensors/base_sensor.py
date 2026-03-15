from abc import ABC, abstractmethod

class BaseSensor(ABC):
    """
    传感器抽象基类
    """

    @abstractmethod
    def read_data(self):
        """
        读取传感器数据
        :return: 原始数据流或解析后的字典
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        健康检查
        :return: True 表示设备在线且正常, False 表示设备断开或异常
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        安全释放硬件资源
        """
        pass