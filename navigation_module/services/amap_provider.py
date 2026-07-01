import requests
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.base_map_api import BaseMapAPI
from config.config_loader import load_config, is_valid_api_key
from utils.logger import get_logger

logger = get_logger(__name__)


class AMapProvider(BaseMapAPI):
    """
    高德地图 API 具体实现类
    """

    def __init__(self):
        config = load_config()
        self.api_key = config['api']['amap']['key']
        self.geocode_url = config['api']['amap']['geocode_url']
        self.walking_url = config['api']['amap']['walking_url']
        self.geo_url = config['api']['amap']['geo_url']

    def _key_ok(self) -> bool:
        if not is_valid_api_key(self.api_key):
            logger.error("API Key 未设置或仍为占位符, 请设置 AMAP_API_KEY 环境变量")
            return False
        return True

    def get_location_name(self, lon: float, lat: float) -> str:
        """调用高德逆地理编码 API"""
        if not self._key_ok():
            return "未知位置"

        params = {
            'key': self.api_key,
            'location': f"{lon},{lat}",
            'radius': 200,
            'extensions': 'base'
        }

        try:
            response = requests.get(self.geocode_url, params=params, timeout=3.0)
            response.raise_for_status()

            data = response.json()
            if data['status'] == '1' and data['regeocode']:
                formatted_address = data['regeocode']['formatted_address']
                return formatted_address
            else:
                logger.warning(f"高德 API 返回错误: {data.get('info')}")
                return "未知位置"

        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {e}")
            return "网络未连接"

    def get_walking_route(self, start_lon: float, start_lat: float,
                          end_lon: float, end_lat: float) -> dict:
        """调用高德步行路径规划 API"""
        if not self._key_ok():
            return None

        params = {
            'key': self.api_key,
            'origin': f"{start_lon},{start_lat}",
            'destination': f"{end_lon},{end_lat}",
        }

        try:
            response = requests.get(self.walking_url, params=params, timeout=5.0)
            response.raise_for_status()

            data = response.json()
            if data['status'] == '1' and data['route']['paths']:
                path = data['route']['paths'][0]

                route_info = {
                    "distance_meters": int(path['distance']),
                    "duration_seconds": int(path['duration']),
                    "steps": path['steps'],
                }
                return route_info
            else:
                logger.warning(f"路径规划失败: {data.get('info')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"路径规划网络请求失败: {e}")
            return None

    def get_coordinate_by_name(self, address_name: str, city: str = "") -> tuple:
        """调用高德正向地理编码 API"""
        if not self._key_ok():
            return None, None

        params = {
            'key': self.api_key,
            'address': address_name,
        }
        if city:
            params['city'] = city

        try:
            response = requests.get(self.geo_url, params=params, timeout=3.0)
            response.raise_for_status()

            data = response.json()
            if data['status'] == '1' and data.get('geocodes'):
                location_str = data['geocodes'][0]['location']
                lon_str, lat_str = location_str.split(',')
                return float(lon_str), float(lat_str)
            else:
                logger.warning(f"无法找到地址 '{address_name}' 的坐标: {data.get('info')}")
                return None, None

        except requests.exceptions.RequestException as e:
            logger.error(f"地名查询网络请求失败: {e}")
            return None, None