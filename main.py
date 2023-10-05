import streamlit as st
import os
import cv2
import numpy as np

# Constants
MIN_GREEN_HUE = 45
MAX_GREEN_HUE = 77
MIN_GREEN_SAT = 19
MAX_GREEN_SAT = 255
MIN_GREEN_VAL = 164
MAX_GREEN_VAL = 255

KERNEL_SIZE = (2, 2)
ERODE_ITERATIONS = 2
DILATE_ITERATIONS = 4
AREA_THRESHOLD = 500  # Minimum contour area to consider as a plant


def process_image(src_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to HSV color space
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color
    lower_green = np.array([MIN_GREEN_HUE, MIN_GREEN_SAT, MIN_GREEN_VAL])
    upper_green = np.array([MAX_GREEN_HUE, MAX_GREEN_SAT, MAX_GREEN_VAL])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Create a kernel for erosion and dilation
    kernel = np.ones(KERNEL_SIZE, np.uint8)

    # Apply morphological operations
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    masked = cv2.threshold(masked, 5, 255, cv2.THRESH_BINARY)[1]
    masked = cv2.erode(masked, kernel, iterations=ERODE_ITERATIONS)
    masked = cv2.dilate(masked, kernel, iterations=DILATE_ITERATIONS)

    # Find contours
    contours, _ = cv2.findContours(
        masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process and count the contours
    plants_number = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > AREA_THRESHOLD:
            plants_number += 1

    return plants_number


def main():
    st.title("Plant Counting Web App")

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        plants_count = process_image(image)

        st.success(f"Total number of plants: {plants_count}")


if __name__ == "__main__":
    main()
