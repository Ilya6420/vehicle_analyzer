import argparse
import os
import requests


# API endpoint
API_URL = "http://localhost:8000/api/v1/analyze-image"


def analyze_vehicle_image(image_path):
    """
    Send an image to the vehicle analyzer API and get the analysis results.

    Parameters:
        image_path: Path to the image file to analyze

    Returns:
        Analysis results from the API
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, 'rb') as image_file:
        files = {'image': (os.path.basename(image_path), image_file, 'image/jpeg')}
        response = requests.post(API_URL, files=files)
        response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze vehicle images using the Vehicle Analyzer API')
    parser.add_argument('--image', type=str, required=True, help='Path to the vehicle image to analyze')
    args = parser.parse_args()

    try:
        results = analyze_vehicle_image(args.image)
        print("Analysis Results:")
        print(results)
    except Exception as e:
        print(f"Error: {e}")
