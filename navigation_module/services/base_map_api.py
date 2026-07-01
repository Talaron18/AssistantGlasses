from abc import ABC, abstractmethod

class BaseMapAPI(ABC):

    @abstractmethod
    def get_location_name(self, lon: float, lat: float) -> str:
        pass

    @abstractmethod
    def get_walking_route(self, start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> dict:
        pass
    
    @abstractmethod
    def get_coordinate_by_name(self, address_name: str, city: str = "") -> tuple:
        pass