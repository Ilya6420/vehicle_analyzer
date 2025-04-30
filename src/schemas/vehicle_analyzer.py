from pydantic import BaseModel, Field


class VehicleAnalysisResponse(BaseModel):
    """
    Response model for vehicle image analysis.
    """
    total_cars: int = Field(
        description="Total number of cars detected in the image",
        example=3
    )
    red_cars: int = Field(
        description="Number of red cars detected in the image",
        example=1
    )
    description: str = Field(
        description="Generated description of the image content",
        example="The image shows a parking lot with three cars. One red sedan, one blue SUV, and one black truck."
    )
