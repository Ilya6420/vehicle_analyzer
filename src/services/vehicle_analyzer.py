import logging

import cv2
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import torch

from .interfaces import ICarDetector, IColorDetector, IImageDescriptionGenerator
from .pipelines import (CPUCarDetector,
                        CPUColorIdentificator,
                        CPUDescriptionGenerator,
                        GPUCarDetector,
                        GPUColorIdentificator,
                        GPUDescriptionGenerator)
from ..schemas.vehicle_analyzer import VehicleAnalysisResponse


logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory class for creating appropriate detection and analysis components based on hardware availability.
    """
    @staticmethod
    def get_car_detector() -> ICarDetector:
        """
        Create and return an appropriate car detector based on hardware availability.
        """
        detector = GPUCarDetector() if torch.cuda.is_available() else CPUCarDetector()
        logger.info(f"Using {detector.__class__.__name__} for car detection")
        return detector

    @staticmethod
    def get_color_identifier() -> IColorDetector:
        """
        Create and return an appropriate color identifier based on hardware availability.
        """
        detector = GPUColorIdentificator() if torch.cuda.is_available() else CPUColorIdentificator()
        logger.info(f"Using {detector.__class__.__name__} for color identification")
        return detector

    @staticmethod
    def get_description_generator() -> IImageDescriptionGenerator:
        """
        Create and return an appropriate image description generator based on hardware availability.
        """
        generator = GPUDescriptionGenerator() if torch.cuda.is_available() else CPUDescriptionGenerator()
        logger.info(f"Using {generator.__class__.__name__} for description generation")
        return generator


class VehicleAnalyzerService:
    """
    Service for analyzing vehicle images to detect cars, identify colors, and generate descriptions.
    """
    def __init__(self,
                 car_detector: ICarDetector,
                 color_identifier: IColorDetector,
                 description_generator: IImageDescriptionGenerator):
        """
        Initialize the vehicle analyzer service with required components.
        
        Parameters:
            car_detector: Component responsible for detecting cars in images
            color_identifier: Component responsible for identifying car colors
            description_generator: Component responsible for generating image descriptions
        """
        self.car_detector = car_detector
        self.color_identifier = color_identifier
        self.description_generator = description_generator
        logger.info("VehicleAnalyzerService initialized with all required components")

    async def analyze_image(self, image: UploadFile) -> VehicleAnalysisResponse:
        """
        Analyze a vehicle image to detect cars, count red cars, and generate a description.
        
        Parameters:
            image: The uploaded image file to analyze
   
        Returns:
            A VehicleAnalysisResponse containing the total car count, red car count, and image description
        """
        try:
            logger.info(f"Processing image: {image.filename}")
            contents = await image.read()
            arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            logger.info(f"Image loaded successfully, dimensions: {img.shape}")

            logger.info("Starting car detection...")
            total = self.car_detector.detect(img)
            bboxes = self.car_detector.get_bboxes(img)
            logger.info(f"Detected {total} cars in the image")

            logger.info("Starting color analysis...")
            red = self.color_identifier.count_red(img, bboxes)
            logger.info(f"Found {red} red cars")

            logger.info("Generating image description...")
            desc = self.description_generator.generate(img)
            logger.info("Description generated successfully")

            logger.info("Analysis completed successfully")
            return VehicleAnalysisResponse(total_cars=total, red_cars=red, description=desc)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": str(e)})
