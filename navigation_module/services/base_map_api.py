from abc import ABC, abstractmethod

class BaseMapAPI(ABC):
    """
    地图 API 抽象基类
    """

    @abstractmethod
    def get_location_name(self, lon: float, lat: float) -> str:
        """
        逆地理编码：将经纬度转换为人类可读的地址描述
        :param lon: 经度
        :param lat: 纬度
        :return: 地址字符串
        """
        pass

    @abstractmethod
    def get_walking_route(self, start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> dict:
        """
        步行路径规划
        :param start_lon: 起点经度
        :param start_lat: 起点纬度
        :param end_lon: 终点经度
        :param end_lat: 终点纬度
        :return: 包含总距离、预计时间、以及每个路段导航指令的字典
        """
        pass
    
    @abstractmethod
    def get_coordinate_by_name(self, address_name: str, city: str = "") -> tuple:
        """
        正向地理编码：将地名转换为经纬度坐标
        :param address_name: 目的地名称 (如 "天安门", "夫子庙")
        :param city: 城市名称 (可选，有助于提高准确率，如 "南京")
        :return: (经度, 纬度) 的元组，如果失败返回 (None, None)
        """
        pass