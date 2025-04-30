from fastapi import APIRouter, UploadFile, File

from ..core.logging import get_logger
from ..schemas.vehicle_analyzer import VehicleAnalysisResponse
from ..services.vehicle_analyzer import PipelineFactory, VehicleAnalyzerService


logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["vehicle-analyzer"])

logger.info("Initializing VehicleAnalyzerService with detectors...")
service = VehicleAnalyzerService(
    PipelineFactory.get_car_detector(),
    PipelineFactory.get_color_identifier(),
    PipelineFactory.get_description_generator()
)


@router.post("/analyze-image", response_model=VehicleAnalysisResponse)
async def analyze_image(image: UploadFile = File(...)):
    """
    Analyze a vehicle image to detect cars and their properties.

    Parameters:
        image: UploadFile - The image file to be analyzed

    Returns:
        Contains total number of cars detected,
        count of red cars, and a description of the image
    """
    logger.info(f"Received request to analyze image: {image.filename}")
    result = await service.analyze_image(image)
    logger.info("Image analysis request completed")
    return result
