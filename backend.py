import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import pandas as pd
import os
import time
import re
from collections import defaultdict
from scipy.spatial import distance as dist
from google import genai
from google.genai import types
import PIL.Image

# Initialize Gemini client
client = genai.Client(api_key="")

detected_data = []

# Load YOLOv8 model
model = YOLO("license_plate_detector.pt")

# Initialize EasyOCR for text recognition
reader = easyocr.Reader(["en"])

# Define HSV color ranges for plate colors
color_ranges = {
    "Red": ([0, 50, 50], [10, 255, 255]),
    "Green": ([35, 50, 50], [85, 255, 255]),
    "Yellow": ([20, 50, 50], [35, 255, 255]),
    "White": ([0, 0, 200], [180, 50, 255]),
    "Black": ([0, 0, 0], [180, 255, 50]),
}

# Create necessary directories
output_image_dir = "detected_plates_images"
csv_filename = "detected_plates.csv"
os.makedirs(output_image_dir, exist_ok=True)

# Store detected plates to prevent duplicates
detected_plates = defaultdict(lambda: None)

# Frame skipping for performance
FRAME_SKIP = 5  # Increase frame skipping to reduce load
frame_count = 0

# Delay between Gemini API requests
GEMINI_DELAY = 2  # Delay in seconds

class CentroidTracker:
    """Tracks detected number plates to reduce redundant detections."""
    def __init__(self, maxDisappeared=10):  # ✅ Fix: Corrected `__init__`
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            inputCentroids[i] = (cX, cY)

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        if len(self.objects) > 0:
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        else:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])

        return self.objects

tracker = CentroidTracker()

def detect_color(plate_img):
    """Detects the dominant color of the number plate."""
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if np.sum(mask) > 5000:
            return color
    return "Unknown"

def save_plate_image(plate_img, plate_text):
    """Saves detected plate image with a timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"plate_{plate_text.replace(' ', '')}{timestamp}.jpg"
    filepath = os.path.join(output_image_dir, filename)
    cv2.imwrite(filepath, plate_img)
    return filename

def standardize_plate_format(plate_text):
    """Formats the number plate to match Indian standards."""
    # Remove any non-alphanumeric characters
    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())

    # Try to match the standard Indian format
    pattern = r'([A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4})'
    match = re.search(pattern, plate_text)

    if match:
        return match.group(0)
    return plate_text

def analyze_plate_with_gemini(image_path):
    """Analyze the plate image using Google Gemini API."""
    image = PIL.Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["What is the number plate from this image! and also the color of the number plate", image]
    )
    return response.text

def process_frame(frame, detected_data):
    """Detects plates, extracts text, classifies color, and tracks objects."""
    global frame_count

    # Skip frames for performance
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        return frame, []

    frame_count += 1

    # Create a copy of the frame for drawing
    display_frame = frame.copy()
    new_detections = []

    # Run YOLOv8 model for detection
    results = model(frame)
    rects = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            rects.append((x1, y1, x2, y2))

    # Update tracker
    tracked_objects = tracker.update(rects)

    for i, (x1, y1, x2, y2) in enumerate(rects):
        # Skip if coordinates are invalid
        if y2 <= y1 or x2 <= x1 or y1 < 0 or x1 < 0:
            continue

        # Extract plate image
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        # Enhance image for better OCR
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Perform OCR
        ocr_results = reader.readtext(thresh)
        plate_text = ' '.join([text[1] for text in ocr_results]).strip()

        # Skip if no text detected
        if not plate_text:
            continue

        # Standardize plate format
        plate_text = standardize_plate_format(plate_text)

        # Detect plate color
        plate_color = detect_color(plate_img)

        # Avoid multiple detections within 5 seconds
        if detected_plates[plate_text] is not None and time.time() - detected_plates[plate_text] < 5:
            # Still draw the rectangle but don't add to detections
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{plate_text} ({plate_color})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            continue

        # Mark as detected
        detected_plates[plate_text] = time.time()

        # Save the plate image
        image_filename = save_plate_image(plate_img, plate_text)
        image_path = os.path.join(output_image_dir, image_filename)

        # Analyze the plate image using Google Gemini API
        gemini_response = analyze_plate_with_gemini(image_path)
        print(f"Gemini Response: {gemini_response}")

        # Add a delay to avoid hitting the rate limit
        time.sleep(GEMINI_DELAY)

        # Add to detection data
        detection = {
            "Plate Text": plate_text,
            "Color": plate_color,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Image Filename": image_filename,
            "Gemini Analysis": gemini_response
        }

        detected_data.append(detection)
        new_detections.append(detection)

        # Draw on the frame
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{plate_text} ({plate_color})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Update the CSV file
    if detected_data:
        df = pd.DataFrame(detected_data)
        df = df.drop_duplicates(subset=["Plate Text"])
        df.to_csv(csv_filename, index=False)

    return display_frame, new_detections

def main():
    detected_data = []

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 30  # Set FPS manually (Webcam FPS might be unreliable)

    # Define video writer
    output_video_path = "output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Recording... Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Process the frame (if needed)
            processed_frame = frame  # Directly use the frame if no modifications

            # Write to video
            out.write(processed_frame)

            # Show live video (optional)
            cv2.imshow("License Plate Detection", processed_frame)

            # Stop recording on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved as {output_video_path}")

if __name__ == "__main__":
    main()
