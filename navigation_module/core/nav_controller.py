import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.gnss.mock_reader import MockGNSSReader
from sensors.gnss.nmea_parser import NMEAParser
from algo.fusion.linear_kalman import LinearKalmanFilter
from algo.geo.coord_transform import CoordTransformer
from algo.geo.haversine import haversine_distance
from services.amap_provider import AMapProvider
from config.config_loader import load_config
from sensors.gnss.serial_reader import GNSSSerialReader
from utils.logger import get_logger

logger = get_logger(__name__)

class NavController:
    """
    盲人智能拐杖导航系统的核心调度器
    """
    def __init__(self):
        logger.info("正在初始化核心控制器...")
        
        # 底层感知与算法
        #self.reader = MockGNSSReader(mock_file_name="nmea_sample.txt")
        self.reader = GNSSSerialReader() # 连接物理串口读取器
        self.parser = NMEAParser()
        self.kalman = LinearKalmanFilter()
        self.transformer = CoordTransformer()
        
        # 云端服务
        self.map_api = AMapProvider()
        
        # 加载系统配置
        config = load_config()
        self.broadcast_distances = config['navigation']['broadcast_distances'] # [50, 10, 3]
        
        self.is_running = False
        
        # 导航状态变量
        self.is_navigating = False
        self.current_route = None
        
        # 模拟从语音模块接收到的自然语言指令
        self.target_name = "北京理工大学良乡校区徐特立图书馆" 
        self.target_lon = None  # 初始坐标为空，等待云端解析
        self.target_lat = None

    def run_main_loop(self):
        """核心大循环"""
        self.is_running = True
        logger.info("主循环已启动，开始守护盲人出行！")
        
        while self.is_running:
            time.sleep(0.01) # 防止 CPU 100%
            
            # 获取原始数据
            raw_line = self.reader.read_data()
            if not raw_line:
                continue
                
            # 解析数据
            parsed_data = self.parser.parse(raw_line)
            if not parsed_data or parsed_data.get('type') != 'RMC' or not parsed_data.get('is_valid'):
                continue
                
            # 卡尔曼滤波平滑
            self.kalman.update((
                parsed_data['longitude'], 
                parsed_data['latitude'], 
                parsed_data['speed_kmh'], 
                parsed_data['true_course']
            ))
            smooth_lon, smooth_lat = self.kalman.get_state()
            
            # 转换中国地图坐标系
            gcj_lon, gcj_lat = self.transformer.wgs84_to_gcj02(smooth_lon, smooth_lat)
            
            # 导航与播报判断
            
            # 假设盲人按下了导航按钮，但还没规划过路线
            if not self.is_navigating:
                # 将语音指令翻译成经纬度
                if not self.target_lon or not self.target_lat:
                    logger.info(f"正在将语音指令 '{self.target_name}' 转换为经纬度坐标...")
                    self.target_lon, self.target_lat = self.map_api.get_coordinate_by_name(self.target_name)
                    
                    if not self.target_lon:
                        logger.error(f"找不到 '{self.target_name}' 的位置，请盲人重新语音输入！")
                        time.sleep(2)
                        continue
                logger.info("开始向高德云端请求路径规划...")
                self.current_route = self.map_api.get_walking_route(
                    gcj_lon, gcj_lat, self.target_lon, self.target_lat
                )
                if self.current_route:
                    logger.info(f"路线获取成功！总距离 {self.current_route['distance_meters']} 米。")
                    logger.info(f"第一步指引: {self.current_route['steps'][0]}")
                    self.is_navigating = True
                else:
                    logger.error("路径规划失败，请检查网络！")
                    time.sleep(2)
                    continue
            
            # 如果正在导航中，实时计算距离目的地的直线距离
            if self.is_navigating:
                distance_to_target = haversine_distance(gcj_lon, gcj_lat, self.target_lon, self.target_lat)
                logger.info(f"当前坐标: ({gcj_lon:.6f}, {gcj_lat:.6f}) | 距目的地还有: {distance_to_target:.1f} 米")
                
                # 梯次播报逻辑测试 (50米, 10米, 3米)
                for threshold in self.broadcast_distances:
                    # 允许有 0.5 米的误差范围，防止刚好错过
                    if abs(distance_to_target - threshold) < 0.5:
                        logger.warning(f"[语音模块播报]: 距离终点还有 {threshold} 米！")
                        
                if distance_to_target <= 1.5:
                    logger.warning("[语音模块播报]: 您已到达目的地附近，导航结束。")
                    self.is_navigating = False
                    self.is_running = False # 演示完毕，退出循环

    def shutdown(self):
        self.is_running = False
        if self.reader:
            self.reader.close()
        logger.info("控制器已安全释放资源。")