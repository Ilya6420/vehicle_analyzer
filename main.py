import os
os.makedirs("models", exist_ok=True)
os.environ["HF_HOME"] = "models"

from fastapi import FastAPI

from src.routers import vehicle_analyzer, utils


app = FastAPI(
    title="Vehicle Analyzer API",
    description="API for analyzing vehicles in images",
    version="1.0.0"
)


app.include_router(vehicle_analyzer.router)
app.include_router(utils.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
