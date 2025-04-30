import abc

import numpy as np


class ICarDetector(abc.ABC):
    """
    Interface for car detection in images.
    """
    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> int:
        """
        Detect the number of cars in an image.
        """
        pass

    @abc.abstractmethod
    def get_bboxes(self, image: np.ndarray) -> list[np.ndarray] | None:
        """
        Get bounding boxes for cars in an image.
        """
        pass


class IColorDetector(abc.ABC):
    """
    Interface for detecting colors of cars in images.
    """
    @abc.abstractmethod
    def count_red(self, image: np.ndarray, bboxes: list[np.ndarray] | None) -> int:
        """
        Count the number of red cars in an image.
        """
        pass


class IImageDescriptionGenerator(abc.ABC):
    """
    Interface for generating textual descriptions of vehicle images.
    """
    @abc.abstractmethod
    def generate(self, image: np.ndarray) -> str:
        """
        Generate a descriptive text for an image containing vehicles.
        """
        pass
