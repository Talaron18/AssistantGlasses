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

    LOW_SPEED_KMH: float = 1.0
    LOW_SPEED_VEL_VAR: float = 100.0

    def __init__(self):
        super().__init__()
        self.R_EARTH = 6371000.0

        self.X = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0
        self.F = np.eye(4)

        self._H_POS = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=float)
        self._H_VEL = np.array([[0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=float)

        self.Q = np.eye(4) * 1e-3

        self.base_pos_var = 5.0
        self.base_vel_var = 1.0

        self._hdop = 1.0

        self.origin_lon = None
        self.origin_lat = None


    def set_hdop(self, hdop: float):
        """注入最近一帧 GGA 解析出的 HDOP, 用于位置观测噪声自适应。"""
        if hdop and hdop > 0:
            self._hdop = float(hdop)


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
        self.P = np.eye(4) * 1.0
        self.is_initialized = True
        logger.info(f"卡尔曼滤波器已初始化: 投影原点 ({lon:.6f}, {lat:.6f})")

    def predict(self, dt: float):
        """
        预测阶段： X_predict = F * X_prev

        """
        if not self.is_initialized:
            return
        if dt <= 0:
            return

        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement: tuple):
        """
        两阶段观测融合:
          Pass 1 — 位置观测 (_H_POS 2×4), 始终执行;
          Pass 2 — 速度观测 (_H_VEL 2×4), 仅在有效行走速度时执行。
        :param measurement: (经度, 纬度, 速度km/h, 航向角)
        """
        if len(measurement) != 4:
            logger.error(f"观测维度异常, 需要4维, 当前为: {len(measurement)}")
            return

        lon, lat, speed_kmh, course = measurement

        if not self.is_initialized:
            self.initialize(lon, lat)
            return

        x, y = self._latlon_to_xy(lon, lat)
        speed_ms = speed_kmh / 3.6
        rad_course = math.radians(course)
        v_x = speed_ms * math.sin(rad_course)
        v_y = speed_ms * math.cos(rad_course)

        pos_var = self.base_pos_var * max(1.0, self._hdop) ** 2
        I4 = np.eye(4)

        Z_p = np.array([[x], [y]])
        R_p = np.diag([pos_var, pos_var])
        S_p = self._H_POS @ self.P @ self._H_POS.T + R_p
        K_p = self.P @ self._H_POS.T @ np.linalg.inv(S_p)
        self.X = self.X + K_p @ (Z_p - self._H_POS @ self.X)
        self.P = (I4 - K_p @ self._H_POS) @ self.P

        if speed_kmh >= self.LOW_SPEED_KMH:
            Z_v = np.array([[v_x], [v_y]])
            R_v = np.diag([self.base_vel_var, self.base_vel_var])
            S_v = self._H_VEL @ self.P @ self._H_VEL.T + R_v
            K_v = self.P @ self._H_VEL.T @ np.linalg.inv(S_v)
            self.X = self.X + K_v @ (Z_v - self._H_VEL @ self.X)
            self.P = (I4 - K_v @ self._H_VEL) @ self.P

    def get_state(self) -> tuple:
        """
        输出滤波后的平滑坐标
        """
        if not self.is_initialized:
            return 0.0, 0.0

        x, y = float(self.X[0, 0]), float(self.X[1, 0])
        return self._xy_to_latlon(x, y)