import os
import glob
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import cv2

# Constants
HOME = os.getcwd()
SOURCE_VIDEO_PATH = f"{HOME}/tea-test-final.mp4"
TARGET_VIDEO_PATH = f"{HOME}/tea-test-final-masked.mp4"

# Color configurations
COLORS = sv.ColorPalette.from_hex(
    ["#808080", "#b8860b", "#0000ff"]
)  # gray, yellow, blue
SPILLING_COLORS = sv.ColorPalette.from_hex(
    ["#808080", "#ff0000", "#0000ff"]
)  # gray, red, blue


mask_annotator = sv.MaskAnnotator(color=COLORS)
box_annotator = sv.BoxAnnotator(thickness=10)

label_annotator_spilling = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_position=sv.Position.CENTER,
    color=SPILLING_COLORS,
    text_scale=6,
    text_thickness=2,
    smart_position=False,
)

label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_position=sv.Position.TOP_CENTER,
    color=COLORS,
    text_scale=2,
    text_thickness=2,
    smart_position=False,
)

tea_annotator = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_position=sv.Position.TOP_CENTER,
    color=COLORS,
    text_scale=2,
    text_thickness=2,
    smart_position=False,
)

rim_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK,
    text_position=sv.Position.TOP_CENTER,
    color=COLORS,
    text_scale=2,
    text_thickness=2,
    smart_position=False,
)


def find_highest_point(mask_array: np.ndarray) -> Optional[Tuple[float, float]]:
    """Find the highest point in a mask array."""
    if len(mask_array) == 0:
        return None
    highest_idx = np.argmin(mask_array[:, 1])
    return (mask_array[highest_idx, 0], mask_array[highest_idx, 1])


def get_peak_points(masks) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Extract rim and tea peak points from masks."""
    peaks = np.zeros((0, 2))
    for mask in masks:
        peaks = np.vstack((peaks, find_highest_point(mask.xy[0])))
    rim_peak, tea_peak = (
        (peaks[1], peaks[0]) if peaks[0][1] > peaks[1][1] else (peaks[0], peaks[1])
    )
    return rim_peak, tea_peak


def is_tea_spilling(masks) -> bool:
    """Determine if tea is about to spill based on mask positions."""
    if len(masks) <= 2:
        return False
    rim_peak, tea_peak = get_peak_points(masks)
    return rim_peak[1] + 100 > tea_peak[1]


def process_frames_from_folder(folder_path: str, model: YOLO):
    """Process individual frames from a folder."""
    for image in sorted(glob.glob(folder_path)):
        image = Image.open(image)
        result = model.predict(image, conf=0.75)[0]

        detections = sv.Detections.from_ultralytics(result)
        annotated_image = image.copy()

        mask_annotator.annotate(annotated_image, detections=detections)
        label_annotator.annotate(annotated_image, detections=detections)

        rim_peak, tea_peak = get_peak_points(result.masks)

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
        sv.plot_images_grid([image, annotated_image], grid_size=(1, 2), size=(10, 10))

        if rim_peak[1] + 100 > tea_peak[1]:
            print("Tea about to spill over")
            break


def process_video(
    model: YOLO,
    source_video_path=SOURCE_VIDEO_PATH,
    target_video_path=TARGET_VIDEO_PATH,
):
    """Process video and create annotated output."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    frame = next(iter(frames_generator))
    result = model.predict(frame, conf=0.75)[0]
    frame_count = 0

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(frames_generator, total=video_info.total_frames):
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1
            annotated_frame = frame.copy()

            if frame_count % video_info.fps == 0:
                result = model.predict(frame, conf=0.75)[0]

            detections = sv.Detections.from_ultralytics(result)
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = tea_annotator.annotate(
                scene=annotated_frame, detections=detections[detections.class_id == 2]
            )
            annotated_frame = rim_annotator.annotate(
                scene=annotated_frame, detections=detections[detections.class_id == 1]
            )

            if result.masks is not None and is_tea_spilling(result.masks):
                annotated_frame = label_annotator_spilling.annotate(
                    scene=annotated_frame,
                    detections=detections[detections.class_id == 1],
                    labels=["Spill Alert"],
                )
                print("Tea about to spill over")

            sink.write_frame(annotated_frame)


def main():
    model = YOLO(f"{HOME}/models/Epoch-500-best-yolo11-segmentation.pt")
    source_video_path = f"{HOME}/tea-test-final.mp4"
    target_video_path = f"{HOME}/tea-test-final-masked.mp4"
    process_video(model, source_video_path, target_video_path)


if __name__ == "__main__":
    main()
