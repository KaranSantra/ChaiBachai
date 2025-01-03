from ultralytics import YOLO
from PIL import Image
import os
import supervision as sv
import glob
import numpy as np

HOME = os.getcwd()
print(HOME)

model = YOLO(f"{HOME}/models/Epoch-500-best-yolo11-segmentation.pt")

colors = sv.ColorPalette.from_hex(
    ["#808080", "#ffff00", "#0000ff"]
)  # gray, yellow, blue
mask_annotator = sv.MaskAnnotator(color=colors)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK, text_position=sv.Position.CENTER, color=colors
)


def find_highest_point(mask_array):
    if len(mask_array) == 0:
        return None
    highest_idx = np.argmin(mask_array[:, 1])
    return (mask_array[highest_idx, 0], mask_array[highest_idx, 1])


for image in sorted(glob.glob(f"{HOME}/SAM-TRAIN-IMAGES/frames-tea-3/*.jpg")):
    image = Image.open(image)
    result = model.predict(image, conf=0.75)[0]
    # print("result.names",result.names)
    detections = sv.Detections.from_ultralytics(result)
    annotated_image = image.copy()
    mask_annotator.annotate(annotated_image, detections=detections)
    label_annotator.annotate(annotated_image, detections=detections)
    peaks = np.zeros((0, 2))
    for mask in result.masks:
        peaks = np.vstack((peaks, find_highest_point(mask.xy[0])))
    rim_peak, tea_peak = (
        (peaks[1], peaks[0]) if peaks[0][1] > peaks[1][1] else (peaks[0], peaks[1])
    )
    point_detection = sv.Detections(
        xyxy=np.array(
            [
                [rim_peak[0], rim_peak[1], rim_peak[0] + 1, rim_peak[1] + 1],
                [tea_peak[0], tea_peak[1], tea_peak[0] + 1, tea_peak[1] + 1],
            ]
        ),
        class_id=np.array([1, 2]),
        confidence=np.array([1.0, 1.0]),
    )
    label_annotator.annotate(
        annotated_image, detections=point_detection, labels=["Rim Peak", "Tea Peak"]
    )
    print("rim_peak", rim_peak)
    print("tea_peak", tea_peak)
    sv.plot_images_grid([image, annotated_image], grid_size=(1, 2), size=(10, 10))
    if rim_peak[1] + 100 > tea_peak[1]:
        print("Tea about to spill over")
        break
    print("\n")
