from abc import ABC, abstractmethod

class BaseSensor(ABC):

    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass
    
    @abstractmethod
    def close(self):
        pass