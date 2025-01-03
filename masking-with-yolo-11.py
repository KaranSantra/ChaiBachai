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
    ["#808080", "#b8860b", "#0000ff"]
)  # gray, yellow, blue

spilling_colors = sv.ColorPalette.from_hex(
    ["#808080", "#ff0000", "#0000ff"]
)  # gray, yellow, blue
mask_annotator = sv.MaskAnnotator(color=colors)
box_annotator = sv.BoxAnnotator(thickness=10)
label_annotator_spillling = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_position=sv.Position.CENTER,
    color=spilling_colors,
    text_scale=6,
    text_thickness=2,
    smart_position=False,
)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_position=sv.Position.TOP_CENTER,
    color=colors,
    text_scale=2,
    text_thickness=2,
    smart_position=False,
)
tea_annotator = sv.LabelAnnotator(
    text_color=sv.Color.WHITE,
    text_position=sv.Position.TOP_CENTER,
    color=colors,
    text_scale=2,
    text_thickness=2,
    smart_position=False,
)
rim_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK,
    text_position=sv.Position.TOP_CENTER,
    color=colors,
    text_scale=2,
    text_thickness=2,
    smart_position=False,
)


def find_highest_point(mask_array):
    if len(mask_array) == 0:
        return None
    highest_idx = np.argmin(mask_array[:, 1])
    return (mask_array[highest_idx, 0], mask_array[highest_idx, 1])


def process_frames_from_folder(folder_path):
    for image in sorted(glob.glob(folder_path)):
        image = Image.open(image)
        result = model.predict(image, conf=0.75)[0]
        print("result.names", result.boxes)
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


# process_frames_from_folder(f"{HOME}/SAM-TRAIN-IMAGES/frames-tea-3/*.jpg")


def is_tea_spilling(masks):
    peaks = np.zeros((0, 2))
    for mask in masks:
        peaks = np.vstack((peaks, find_highest_point(mask.xy[0])))
    if len(peaks) <= 2:
        return False
    rim_peak, tea_peak = (
        (peaks[1], peaks[0]) if peaks[0][1] > peaks[1][1] else (peaks[0], peaks[1])
    )
    if rim_peak[1] + 100 > tea_peak[1]:
        return True
    return False


from tqdm import tqdm

SOURCE_VIDEO_PATH = f"{HOME}/tea-test-final.mp4"
TARGET_VIDEO_PATH = f"{HOME}/tea-test-final-masked.mp4"


video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frames_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame_iterator = iter(frames_generator)
frame = next(frame_iterator)
frame_count = 0
# open target video
result = model.predict(frame, conf=0.75)[0]
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(frames_generator, total=video_info.total_frames):
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
            # annotated_frame = box_annotator.annotate(
            #     scene=annotated_frame,
            #     detections=detections[detections.class_id == 1],
            # )
            annotated_frame = label_annotator_spillling.annotate(
                scene=annotated_frame,
                detections=detections[detections.class_id == 1],
                labels=["Tea Spilling"],
            )
            print("Tea about to spill over")
        sink.write_frame(annotated_frame)

        # add frame to target video
