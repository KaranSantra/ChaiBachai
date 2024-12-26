import cv2
import os
import glob

input_folder = "dataset-color"
output_folder = "all-images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = glob.glob(os.path.join(input_folder, "**/*.*"), recursive=True)
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

counter = 1
for image_path in image_files:
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension in valid_extensions:
        img = cv2.imread(image_path)
        if img is not None:
            new_filename = f"test-{counter}{file_extension}"
            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, img)
            counter += 1
