from fastapi import APIRouter
from fastapi.responses import JSONResponse


router = APIRouter(tags=["utility"])


@router.get("/")
async def welcome():
    return JSONResponse(
        content={
            "message": "Welcome to the Vehicle Analyzer API!",
            "version": "1.0.0",
            "endpoints": {
                "analyze-image": "/api/v1/analyze-image"
            }
        }
    )
