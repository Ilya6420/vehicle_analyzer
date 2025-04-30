# Image Analysis API

A RESTful API that analyzes images to detect cars, identify red cars, and generate image descriptions.

## Features

- Car detection
- Red car identification
- Image description generation

## Prerequisites

- Python 3.13
- Poetry package manager
- For GPU support:
  - CUDA-compatible GPU with appropriate drivers installed
  - CUDA Toolkit
  - cuDNN 8.5 or higher

**Note**: You don't need to install it all, just the drivers. All the necessary stuff will be installed with pytorch.

## Installation
1. Install poetry using pip:
```bash
pip install poetry==2.1.2
```
2. Install poetry shell plugin
```bash
pip install poetry-plugin-shell==1.0.1
```
3. Clone the repository:
```bash
git clone <repository-url>
cd vehicle_analyzer
```
4. Install dependencies using Poetry:
```bash
poetry install --no-root
```
5. Activate the virtual environment:
```bash
poetry shell
```
## Usage
1. Start the API server:
```bash
python main.py
```
or
```bash
uvicorn main:app
```
2. The API will be available at `http://127.0.0.1:8000/api/v1/analyze-image` or `http://localhost:8000/api/v1/analyze-image`
3. Send a POST request to `api/v1/analyze-image` endpoint with an image file:

Using **curl**:
```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://127.0.0.1:8000/api/v1/analyze-image
```
Using test script:
```bash
python test_endpoint/test_vehicle_analyzer.py --image 'path/to/your/image.jpg'
```
Using Swagger:
```bash
http://127.0.0.1:8000/docs
```
## API Endpoint

### POST /analyze-image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with key "image" containing the image file

**Response:**
```json
{
    "total_cars": 10,
    "red_cars": 3,
    "description": "The image depicts a busy urban street with several vehicles..."
}
```

## Technical Details

### GPU Implementation (Moondream Model)
When running with GPU support, the API uses the Moondream model for all features:
- Car detection and counting
- Red car identification
- Image description generation

This unified approach provides better performance and consistency in results.

**Note**: For car detection and identification we take query mode approach, but we can use point or detection mode and achieve the same results.

#### Design Rationale
The GPU implementation uses a single vision-language model (Moondream) for all tasks to maximize efficiency. This design choice:

- Better results, as we are utilizing good VLM.
- Leverages GPU acceleration for real-time performance
- Consistency as we use only 1 model for all the tasks.

### CPU Implementation (Combined Models)
When running in CPU mode, the API uses a combination of different models and approaches:
- Car Detection: YOLOv8 model for object detection
  - Specifically identifies cars (class 2 in COCO dataset)
- Red Car Identification: Classic Computer Vision approach
  - Converts detected car regions to HSV color space
  - Uses color thresholding to identify red cars
  - A car is considered red if more than 20% of its pixels fall within the red color range
- Image Description: BLIP-2 model for image captioning
  - Generates natural language descriptions of the image content

#### Design Rationale
The CPU implementation uses specialized models for each task to optimize for limited resources. This approach:
- Uses lightweight models that can run efficiently without GPU acceleration
- Employs traditional CV techniques for color detection to reduce computational load
- Provides a fallback solution for environments without GPU availability
- But in general the quality is lower than VLM, espececially for image description.