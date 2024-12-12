import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Real-world scale factor (meters per pixel)
meters_per_pixel = 0.05  # Example value; replace with your actual scale

# Path to the image
image_path = r"../Top veiw img/top view.png"  # Adjust the path if needed

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: File not found at {image_path}")
    exit()

# Load the image
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Image could not be loaded. Please check the file format and path.")
    exit()

try:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a fixed threshold (manually tuned for precision)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to remove noise and close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the largest bounding box
    largest_area = 0
    largest_bbox = None

    # Iterate over contours and filter
    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip small contours
        if area < 5000:  # Adjust this threshold to match the building size
            continue

        # Get bounding box
        rect = cv2.minAreaRect(contour)  # Rotated bounding box
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Use np.int32 instead of np.int0

        # Track the largest bounding box
        if area > largest_area:
            largest_area = area
            largest_bbox = box

    # Calculate the real-world area
    real_world_area = largest_area * (meters_per_pixel ** 2)

    # Draw the largest bounding box on the image
    output_image = image.copy()
    if largest_bbox is not None:
        cv2.drawContours(output_image, [largest_bbox], 0, (0, 255, 0), 3)

    # Convert BGR to RGB for visualization
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display the result and print the area
    plt.figure(figsize=(10, 10))
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()

    print(f"Pixel Area of Bounding Box: {largest_area:.2f} pixels")
    print(f"Real-World Area of Bounding Box: {real_world_area:.2f} square meters")

    # Save the output image (optional)
    output_path = "../Top output/area_calculated_bounding_box_try.jpeg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Output image saved to: {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
