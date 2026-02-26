import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algo.fusion.base_filter import BaseFilter
from utils.logger import get_logger

logger = get_logger(__name__)

class LinearKalmanFilter(BaseFilter):
    """
    2D GPS 轨迹的线性卡尔曼滤波器 (恒速模型)
    状态向量 X = [lon, lat, v_lon, v_lat]^T
    """
    def __init__(self):
        super().__init__()
        
        # 状态向量 X:[经度, 纬度, 经度变化率, 纬度变化率]
        self.X = np.zeros((4, 1))
        
        # 状态协方差矩阵 P: 
        self.P = np.eye(4) * 1.0
        
        # 状态转移矩阵 F 
        self.F = np.eye(4)
        
        # 观测矩阵 H:
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 过程噪声协方差矩阵 Q
        q_noise = 1e-5
        self.Q = np.eye(4) * q_noise
        
        # 观测噪声协方差矩阵 R
        r_noise = 1e-4
        self.R = np.array([
            [r_noise, 0],
            [0, r_noise]
        ])

    def initialize(self, initial_lon: float, initial_lat: float):
        """
        使用获取到的第一个有效 GPS 点初始化滤波器
        """
        self.X = np.array([[initial_lon], [initial_lat], [0.0], [0.0]])
        self.is_initialized = True
        logger.info(f"卡尔曼滤波器已初始化: 初始坐标 ({initial_lon:.6f}, {initial_lat:.6f})")

    def predict(self, dt: float):
        """
        预测阶段： X_predict = F * X_prev
        """
        if not self.is_initialized:
            return

        # 更新状态转移矩阵
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # 状态预测
        self.X = np.dot(self.F, self.X)
        
        # 协方差预测
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement: tuple):
        """
        更新阶段（融入观测值纠偏）
        :param measurement: (观测经度, 观测纬度)
        """
        if not self.is_initialized:
            self.initialize(measurement[0], measurement[1])
            return

        # 观测向量 Z
        Z = np.array([[measurement[0]], [measurement[1]]])

        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态
        Y = Z - np.dot(self.H, self.X) # 误差创新(Innovation)
        self.X = self.X + np.dot(K, Y)

        # 更新协方差 
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self) -> tuple:
        """
        输出滤波后的平滑坐标
        """
        if not self.is_initialized:
            return 0.0, 0.0
        
        # 返回滤波后的 经度, 纬度
        return float(self.X[0, 0]), float(self.X[1, 0])