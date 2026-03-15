import time
import sys
import os
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.gnss.nmea_parser import NMEAParser
from algo.fusion.linear_kalman import LinearKalmanFilter
from algo.geo.coord_transform import CoordTransformer
from algo.geo.haversine import haversine_distance
from services.amap_provider import AMapProvider
from config.config_loader import load_config
from sensors.gnss.serial_reader import GNSSSerialReader
from utils.logger import get_logger

logger = get_logger(__name__)

class NavController(threading.Thread):
    """
    盲人智能拐杖导航系统 (多线程状态机)
    """
    def __init__(self, nav_queue: queue.Queue, tts_queue: queue.Queue):
        # 设为守护线程
        super().__init__(daemon=True) 
        logger.info("初始化核心控制器...")
        
        # 通信队列
        self.nav_queue = nav_queue  
        self.tts_queue = tts_queue  
        
        # 底层感知与算法
        self.reader = GNSSSerialReader()
        self.parser = NMEAParser()
        self.kalman = LinearKalmanFilter()
        self.transformer = CoordTransformer()
        
        # 云端服务
        self.map_api = AMapProvider()
        
        # 加载系统配置
        config = load_config()
        self.broadcast_distances = config['navigation']['broadcast_distances'] # [50, 10, 3]
        
        # 状态机与导航状态变量
        self.state = "IDLE"
        self.target_name = None 
        self.target_lon = None
        self.target_lat = None
        self.current_route = None
        self.current_gcj_lon = None
        self.current_gcj_lat = None

    def run(self):
        """独立线程在后台运行"""
        logger.info("导航后台线程已启动")
        
        while True:
            time.sleep(0.01)
            # 更新坐标
            raw_line = self.reader.read_data()
            
            if raw_line:
                # 解析数据
                parsed_data = self.parser.parse(raw_line)
                if parsed_data and parsed_data.get('type') == 'RMC' and parsed_data.get('is_valid'):
                    # 卡尔曼滤波平滑
                    self.kalman.update((
                        parsed_data['longitude'], 
                        parsed_data['latitude'], 
                        parsed_data['speed_kmh'], 
                        parsed_data['true_course']
                    ))
                    smooth_lon, smooth_lat = self.kalman.get_state()
                    # 转换中国地图坐标系
                    self.current_gcj_lon, self.current_gcj_lat = self.transformer.wgs84_to_gcj02(smooth_lon, smooth_lat)
            
            # 状态机
            # IDLE
            if self.state == "IDLE":
                try:
                    # 查看队列新地名
                    new_target = self.nav_queue.get_nowait()
                    # "结束导航"
                    if new_target == "STOP":
                        logger.info("接收到中止信号，保持 IDLE 状态")
                        self.tts_queue.put("收到，已为您结束本次导航。")
                        self.target_name = None
                        self.current_route = None
                        continue # 下一次循环

                    self.target_name = new_target
                    self.state = "PLANNING" # 收到地名，切到规划状态
                    
                    self.tts_queue.put(f"收到指令，正在为您规划去 {self.target_name} 的路线")
                    logger.info(f"状态切换 -> PLANNING: 目标 {self.target_name}")
                except queue.Empty:
                    pass # 保持静默

            # PLANNING
            elif self.state == "PLANNING":
                logger.info(f"开始向高德云端请求 '{self.target_name}' 的位置和路线...")
                self.target_lon, self.target_lat = self.map_api.get_coordinate_by_name(self.target_name)
                
                if not self.target_lon:
                    logger.error(f"找不到 '{self.target_name}' 的位置")
                    self.tts_queue.put(f"抱歉，在地图上找不到 {self.target_name}，请重新语音输入")
                    self.state = "IDLE"
                    continue
                
                # 请求步行路径规划
                self.current_route = self.map_api.get_walking_route(
                    self.current_gcj_lon, self.current_gcj_lat, self.target_lon, self.target_lat
                )
                
                if self.current_route:
                    dist = self.current_route['distance_meters']
                    step = self.current_route['steps'][0]
                    logger.info(f"路线获取成功, 总距离 {dist} 米")
                    
                    # 播报第一步指引
                    self.tts_queue.put(f"路线规划成功，总距离 {dist} 米。第一步：{step}")
                    self.state = "NAVIGATING"
                    logger.info("状态切换 -> NAVIGATING")
                else:
                    logger.error("路径规划失败")
                    self.tts_queue.put("路径规划失败，可能是网络问题，请重试")
                    self.state = "IDLE"

            # NAVIGATING
            elif self.state == "NAVIGATING":
                # 距离目的地的直线距离
                distance_to_target = haversine_distance(self.current_gcj_lon, self.current_gcj_lat, self.target_lon, self.target_lat)
                
                # 梯次播报
                for threshold in self.broadcast_distances:
                    # 允许有 0.5 米的误差范围，防止刚好错过
                    if abs(distance_to_target - threshold) < 0.5:
                        logger.warning(f"触发播报阈值: {threshold}米")
                        self.tts_queue.put(f"距离目的地还有 {threshold} 米")
                        time.sleep(1) 
                        
                # 到达目的地判定
                if distance_to_target <= 1.5:
                    logger.warning("已到达目的地")
                    self.tts_queue.put("您已到达目的地附近，本次导航结束")
                    self.state = "IDLE"
                    logger.info("状态切换 -> IDLE")

                # 处理导航中途强行中断或变更目的地的情况
                if not self.nav_queue.empty():
                    logger.info("导航中途收到新指令，正在中断当前导航...")
                    self.state = "IDLE"

    def shutdown(self):
        """安全释放资源"""
        if self.reader:
            self.reader.close()
        logger.info("控制器已安全释放资源")