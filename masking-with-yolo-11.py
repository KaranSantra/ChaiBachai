from ultralytics import YOLO
from PIL import Image
import os
import supervision as sv
import glob

HOME = os.getcwd()
print(HOME)

model = YOLO(f"{HOME}/Epoch-500-best-yolo11-segmentation.pt")

colors = sv.ColorPalette.from_hex(
    ["#808080", "#ffff00", "#0000ff"]
)  # gray, yellow, blue
mask_annotator = sv.MaskAnnotator(color=colors)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK, text_position=sv.Position.CENTER, color=colors
)

annotated_images = []
n = 0

for image in glob.glob(f"{HOME}/SAM-TRAIN-IMAGES/frames-tea-2/*.jpg"):
    image = Image.open(image)
    result = model.predict(image, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(result)
    annotated_image = image.copy()
    mask_annotator.annotate(annotated_image, detections=detections)
    label_annotator.annotate(annotated_image, detections=detections)
    sv.plot_image(annotated_image, size=(20, 20))
    annotated_images.extend([annotated_image])
    n += 1

# sv.plot_images_grid(annotated_images, grid_size=(int(n / 2), 2), size=(20, 20))
