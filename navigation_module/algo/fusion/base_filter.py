from abc import ABC, abstractmethod
import numpy as np

class BaseFilter(ABC):
    """
    传感器融合与滤波抽象基类
    """
    def __init__(self):
        self.is_initialized = False

    @abstractmethod
    def predict(self, dt: float):
        """
        状态预测方程
        :param dt: 距离上一次更新的时间间隔（秒）
        """
        pass

    @abstractmethod
    def update(self, measurement: np.ndarray):
        """
        状态更新方程
        :param measurement: 传感器的观测值矩阵
        """
        pass

    @abstractmethod
    def get_state(self) -> tuple:
        """
        获取当前最优估计状态
        :return: (经度, 纬度) 等状态元组
        """
        pass