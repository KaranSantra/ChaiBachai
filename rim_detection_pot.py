"""Rim Detection - Pot

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LlELUJ20MMEABrgHfG_d0FXTTzOEwTjl
"""

import numpy as np
import cv2
from IPython.display import Image as IPyImage, display
import glob
import time
from ultralytics import YOLO
import ultralytics
import os

HOME = os.getcwd()
print(HOME)
ultralytics.checks()


# trainedModel = f"{HOME}/drive/MyDrive/Tea-boiling-over/best-pot-detection-yolo11s-6.pt"
# testFolder = f"{HOME}/drive/MyDrive/Tea-boiling-over/test-images"

trainedModel = f"{HOME}/models/best-pot-detection-yolo11s-6.pt"
testFolder = f"{HOME}/test-images"


class PotRimDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_pots(self, image, conf=0.5, imgsz=256):
        """Detect pots in the image using YOLO model"""
        results = self.model(image, save=True, imgsz=imgsz, conf=conf)
        pot_bboxes = []
        for result in results:
            pot_bboxes.extend(result.boxes.xyxy.tolist())
        return pot_bboxes

    def detect_rim(self, image, boxes):
        """Detect rim for each pot in the image"""
        detections = []
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            # Process ROI
            # roi = self._get_roi(image, x_min, y_min, x_max, y_max)
            norm_img = self._normalize_perspective(image, box)
            cv2.imshow("Rim Edges", norm_img)
            cv2.waitKey(0)
            edges = self._preprocess_image(norm_img)
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
            # rim_edges = self._mask_top_portion(edges, y_max - y_min)
            # cv2.imshow("Rim Edges", rim_edges)
            # cv2.waitKey(0)
            # Detect rim using circles and contours
            rim_line, output_image = self._detect_rim_features(edges, norm_img)
            detections.append((rim_line, output_image))
        return detections

    def _get_roi(self, image, x_min, y_min, x_max, y_max):
        """Extract region of interest"""
        return image[y_min:y_max, x_min:x_max]

    def _preprocess_image(self, roi):
        """Preprocess ROI for edge detection"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 10, 250)

    def _mask_top_portion(self, edges, box_height):
        """Create mask for top portion of pot"""
        mask = np.zeros_like(edges)
        top_mask_height = int(0.3 * box_height)
        mask[:top_mask_height, :] = 255
        return cv2.bitwise_and(edges, edges, mask=mask)

    def _detect_rim_features(self, rim_edges, roi):
        """Detect rim using circles and contours"""
        output_image = roi.copy()
        rim_line = None

        # Try Hough circles first
        circles = cv2.HoughCircles(
            rim_edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=0,
        )

        if circles is not None:
            rim_line = self._process_circles(circles, output_image)

        # Fall back to contours if no circles detected
        if rim_line is None:
            rim_line = self._process_contours(rim_edges, output_image)

        if rim_line is not None:
            cv2.line(
                output_image,
                (0, rim_line),
                (output_image.shape[1], rim_line),
                (0, 0, 255),
                2,
            )

        return rim_line, output_image

    def _process_circles(self, circles, output_image):
        """Process detected circles and draw vertical line from intersection"""
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)

            # Calculate intersection of rim line and circle
            rim_line = int(y - r)
            intersection_x = x

            # Check if intersection point is valid
            if rim_line > 0:
                # Draw vertical line from intersection to bottom of frame
                cv2.line(
                    output_image,
                    (intersection_x, rim_line),
                    (intersection_x, output_image.shape[0]),
                    (255, 0, 0),
                    2,
                )

            return rim_line
        return None

    def _process_contours(self, rim_edges, output_image):
        """Process contours as fallback"""
        contours, _ = cv2.findContours(
            rim_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.drawContours(output_image, [largest_contour], -1, (255, 0, 0), 2)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            return y
        return None

    def _normalize_perspective(self, image, box):
        """Apply perspective transformation to normalize pot ROI"""
        x_min, y_min, x_max, y_max = map(int, box)

        # Approximate pot as a rectangle
        src_pts = np.float32(
            [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
        )
        dst_pts = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

        # Compute homography matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        normalized_roi = cv2.warpPerspective(image, matrix, (300, 300))
        return normalized_roi


def main():
    # Initialize detector
    detector = PotRimDetector(trainedModel)
    # Process images
    latest_folder = max(glob.glob(f"{testFolder}"), key=os.path.getmtime)
    for img_path in glob.glob(f"{latest_folder}/*.*")[:]:
        image = cv2.imread(img_path)
        pot_bboxes = detector.detect_pots(image)

        detections = detector.detect_rim(image, pot_bboxes)

        for rim_line, output_image in detections:
            cv2.imshow("Detected Rim", output_image)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
