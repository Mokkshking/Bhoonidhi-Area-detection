from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np
import os
from PIL import Image
import io

# Create the FastAPI app
app = FastAPI()

# Real-world scale factor (meters per pixel)
METERS_PER_PIXEL = 0.05  # Example value; adjust as needed

# Output directory for processed images
OUTPUT_DIR = "../output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image(image_path):
    """
    Processes the image to find the largest bounding box and calculate the area.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Error loading image. Check the file format and path.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a fixed threshold
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to remove noise and close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the largest bounding box
    largest_area = 0
    largest_bbox = None

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip small contours
        if area < 5000:
            continue

        # Get bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Track the largest bounding box
        if area > largest_area:
            largest_area = area
            largest_bbox = box

    # Calculate the real-world area
    real_world_area = largest_area * (METERS_PER_PIXEL ** 2)

    # Draw the largest bounding box on the image
    output_image = image.copy()
    if largest_bbox is not None:
        cv2.drawContours(output_image, [largest_bbox], 0, (0, 255, 0), 3)

    # Save the output image
    output_path = os.path.join(OUTPUT_DIR, "processed_image.jpg")
    cv2.imwrite(output_path, output_image)

    return largest_area, real_world_area, output_path


@app.post("/process-image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image, process it, and calculate the largest bounding box area.
    """
    try:
        # Save the uploaded file temporarily
        input_image_path = os.path.join(OUTPUT_DIR, file.filename)
        with open(input_image_path, "wb") as f:
            f.write(await file.read())

        # Process the image
        largest_area, real_world_area, output_image_path = process_image(input_image_path)

        # Return the results
        return JSONResponse(
            content={
                "pixel_area": largest_area,
                "real_world_area": real_world_area,
                "output_image_path": f"/download-image/{os.path.basename(output_image_path)}",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-image/{filename}")
def download_image(filename: str):
    """
    Endpoint to download the processed image.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
