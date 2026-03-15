import numpy as np
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algo.fusion.base_filter import BaseFilter
from utils.logger import get_logger

logger = get_logger(__name__)

class LinearKalmanFilter(BaseFilter):
    """
    基于局部笛卡尔坐标系 (近似 ENU) 的线性卡尔曼滤波器
    状态向量 X = [x, y, v_x, v_y]^T (单位: 米, 米/秒)
    """
    def __init__(self):
        super().__init__()
        self.R_EARTH = 6371000.0  # 地球平均半径 (米)

        # 状态向量 X:[x(东向), y(北向), v_x, v_y]
        self.X = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0
        self.F = np.eye(4)
        
        # 观测矩阵 H: 现在能够同时观测位置(x,y)和速度(v_x, v_y)
        self.H = np.eye(4)
        
        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(4) * 1e-3
        
        # 观测噪声协方差矩阵 R 
        self.R = np.diag([5.0, 5.0, 1.0, 1.0])

        # 局部坐标系原点
        self.origin_lon = None
        self.origin_lat = None

    def _latlon_to_xy(self, lon: float, lat: float) -> tuple:
        """将经纬度投影到以原点为中心的局部笛卡尔坐标系 (米)"""
        if self.origin_lon is None:
            return 0.0, 0.0
        rad_lat0 = math.radians(self.origin_lat)
        x = math.radians(lon - self.origin_lon) * math.cos(rad_lat0) * self.R_EARTH
        y = math.radians(lat - self.origin_lat) * self.R_EARTH
        return x, y

    def _xy_to_latlon(self, x: float, y: float) -> tuple:
        """将局部坐标系 (米) 反解回经纬度"""
        if self.origin_lon is None:
            return 0.0, 0.0
        rad_lat0 = math.radians(self.origin_lat)
        lon = self.origin_lon + math.degrees(x / (self.R_EARTH * math.cos(rad_lat0)))
        lat = self.origin_lat + math.degrees(y / self.R_EARTH)
        return lon, lat

    def initialize(self, lon: float, lat: float):
        """设定局部坐标系原点并初始化状态"""
        self.origin_lon = lon
        self.origin_lat = lat
        self.X = np.zeros((4, 1))
        self.is_initialized = True
        logger.info(f"卡尔曼滤波器已初始化: 投影原点 ({lon:.6f}, {lat:.6f})")

    def predict(self, dt: float):
        """
        预测阶段： X_predict = F * X_prev
        """
        if not self.is_initialized:
            return

        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement: tuple):
        """
        融入观测值纠偏
        :param measurement: (经度, 纬度, 速度km/h, 航向角)
        """
        if len(measurement) != 4:
            logger.error(f"观测维度异常, 需要4维, 当前为: {len(measurement)}")
            return

        lon, lat, speed_kmh, course = measurement

        if not self.is_initialized:
            self.initialize(lon, lat)
            return

        # 1. 位置转换 (米)
        x, y = self._latlon_to_xy(lon, lat)

        # 2. 速度向量分解 (米/秒)
        speed_ms = speed_kmh / 3.6
        rad_course = math.radians(course)
        v_x = speed_ms * math.sin(rad_course)  # 东向速度
        v_y = speed_ms * math.cos(rad_course)  # 北向速度

        # 3. 构建观测向量 Z
        Z = np.array([[x], [y], [v_x], [v_y]])

        # 4. 卡尔曼增益与状态更新
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        Y = Z - np.dot(self.H, self.X)
        self.X = self.X + np.dot(K, Y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self) -> tuple:
        """
        输出滤波后的平滑坐标
        """
        if not self.is_initialized:
            return 0.0, 0.0
        
        x, y = float(self.X[0, 0]), float(self.X[1, 0])
        return self._xy_to_latlon(x, y)