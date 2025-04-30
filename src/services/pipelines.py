import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
import torch
from ultralytics import YOLO

from ..core.logging import get_logger
from .interfaces import ICarDetector, IColorDetector, IImageDescriptionGenerator
from ..utils.constants import (MOONDREAM_CAR_DETECTOR_PROMPT,
                               MOONDREAM_COLOR_IDENTIFICATION_PROMPT)


logger = get_logger(__name__)


class MoondreamManager:
    """
    Manager for loading the Moondream vision-language model once.
    """
    _instance = None

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading Moondream2 on {device}")
            cls._instance = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-04-14",
                trust_remote_code=True,
                device_map={"": device}
            )
            logger.info("Moondream model loaded successfully")
        return cls._instance


class GPUCarDetector(ICarDetector):
    """
    Car detector implementation that uses GPU-accelerated vision-language model.
    """
    def __init__(self):
        self.model = MoondreamManager.get_model()
        logger.debug("Initialized GPUCarDetector")

    def detect(self, image: np.ndarray) -> int:
        """
        Detect the number of cars in an image using the Moondream model.
        
        Parameters:
            image: numpy array representing the image
            
        Returns:
            The total count of cars detected
        """
        logger.debug("GPUCarDetector.detect() called")
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        resp = self.model.query(pil, MOONDREAM_CAR_DETECTOR_PROMPT)["answer"]
        try:
            count = int(resp)
            logger.info(f"GPUCarDetector: detected {count} cars")
            return count
        except ValueError:
            logger.warning("GPUCarDetector: failed to parse response")
            return 0

    def get_bboxes(self, image: np.ndarray) -> None:
        # GPU path uses model for count only; no bounding boxes
        # TODO: Implement bb extraction mode for moondream
        return None


class CPUCarDetector(ICarDetector):
    """
    Car detector implementation that uses CPU-based YOLO model.
    """
    def __init__(self):
        logger.debug("Initialized CPUCarDetector with YOLOv8n")
        self.model = YOLO('models/yolov8n.pt')

    def detect(self, image: np.ndarray) -> int:
        """
        Detect the number of cars in an image using YOLOv8n.
        
        Parameters:
            image: numpy array representing the image
            
        Returns:
            The total count of cars detected
        """
        logger.debug("CPUCarDetector.detect() called")
        results = self.model(image)
        count = 0
        for res in results:
            for box in res.boxes:
                if int(box.cls) == 2:
                    count += 1
        logger.info(f"CPUCarDetector: detected {count} cars")
        return count

    def get_bboxes(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Get bounding boxes for cars in an image using YOLOv8n.
        
        Parameters:
            image: numpy array representing the image
            
        Returns:
            List of bounding boxes as numpy arrays
        """
        logger.debug("CPUCarDetector.get_bboxes() called")
        boxes = []
        results = self.model(image)
        for res in results:
            for box in res.boxes:
                if int(box.cls) == 2:
                    boxes.append(box.xyxy[0].cpu().numpy())
        logger.info(f"CPUCarDetector: got {len(boxes)} bounding boxes")
        return boxes


class GPUColorIdentificator(IColorDetector):
    """
    Color detector implementation that uses GPU-accelerated vision-language model.
    """
    def __init__(self):
        self.model = MoondreamManager.get_model()
        logger.debug("Initialized GPUColorIdentificator")

    def count_red(self, image: np.ndarray, bboxes: list[np.ndarray] | None) -> int:
        """
        Count the number of red cars in an image using the Moondream model.
        
        Parameters:
            image: numpy array representing the image
            bboxes: list of bounding boxes (not used in this implementation)
            
        Returns:
            The count of red cars detected
        """
        logger.debug("GPUColorIdentificator.count_red() called")
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        resp = self.model.query(pil, MOONDREAM_COLOR_IDENTIFICATION_PROMPT)["answer"]
        try:
            red_count = int(resp)
            logger.info(f"GPUColorIdentificator: counted {red_count} red cars")
            return red_count

        except ValueError:
            logger.warning("GPUColorIdentificator: failed to parse response")
            return 0


class CPUColorIdentificator(IColorDetector):
    """
    Color detector implementation that uses CPU-based HSV color thresholding.
    """
    def __init__(self):
        logger.debug("Initialized CPUColorIdentificator (color via HSV thresholds)")

    def count_red(self, image: np.ndarray, bboxes: list[np.ndarray]) -> int:
        """
        Count the number of red cars in an image using HSV color thresholding.
        
        Parameters:
            image: numpy array representing the image
            bboxes: list of bounding boxes for detected cars
            
        Returns:
            The count of red cars detected
        """
        logger.debug("CPUColorIdentificator.count_red() called")
        red_count = 0
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            roi = image[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            ranges = [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])]
            mask = None
            for low, high in ranges:
                m = cv2.inRange(hsv, np.array(low), np.array(high))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            if np.sum(mask > 0) / mask.size > 0.2:
                red_count += 1
        logger.info(f"CPUColorIdentificator: counted {red_count} red cars")
        return red_count


class GPUDescriptionGenerator(IImageDescriptionGenerator):
    """
    Image description generator that uses GPU-accelerated vision-language model.
    """
    def __init__(self):
        self.model = MoondreamManager.get_model()
        logger.debug("Initialized GPUDescriptionGenerator")

    def generate(self, image: np.ndarray) -> str:
        """
        Generate a descriptive text for an image using the Moondream model.
        
        Parameters:
            image: numpy array representing the image
            
        Returns:
            A string description of the image content
        """
        logger.debug("GPUDescriptionGenerator.generate() called")
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        caption = self.model.caption(pil, length="short")["caption"]
        logger.info(f"GPUDescriptionGenerator: generated caption '{caption}'")
        return caption


class CPUDescriptionGenerator(IImageDescriptionGenerator):
    """
    Image description generator that uses CPU-based BLIP model.
    """
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        logger.debug("Initialized CPUDescriptionGenerator with BLIP")

    def generate(self, image: np.ndarray) -> str:
        """
        Generate a descriptive text for an image using the BLIP model.
        
        Parameters:
            image: numpy array representing the image
            
        Returns:
            A string description of the image content
        """
        logger.debug("CPUDescriptionGenerator.generate() called")
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(pil, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"CPUDescriptionGenerator: generated caption '{caption}'")
        return caption
