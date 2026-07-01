from abc import ABC, abstractmethod
import numpy as np

class BaseFilter(ABC):
    def __init__(self):
        self.is_initialized = False

    @abstractmethod
    def predict(self, dt: float):
        pass

    @abstractmethod
    def update(self, measurement: np.ndarray):
        pass

    @abstractmethod
    def get_state(self) -> tuple:
        pass