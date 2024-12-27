import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor
import os

HOME = os.getcwd()
print("HOME:", HOME)
LIVE_VIDEO_PATH = "http://192.168.1.202:8080/video"
# !pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
# !pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision==0.23.0
# %pip install "ultralytics<=8.3.40"
# !mkdir -p {HOME}/weights
# !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
import torch
import cv2
import supervision as sv

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    # points_per_side= 16,
    # points_per_batch = 64,
    # pred_iou_thresh=0.55,
    # stability_score_thresh=0.75,
    # stability_score_offset=1.0,
    # box_nms_thresh = 0.7,
    # crop_n_layers = 0,
    # crop_nms_thresh = 0.7,
    # crop_overlap_ratio = 512 / 1500,
    # crop_n_points_downscale_factor= 1,
    # point_grids = None,
    # min_mask_region_area = 0
)


# widget helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


# helper function that loads an image before adding it to the widget
import base64


def encode_image(filepath):
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


IS_COLAB = True
import os

IMAGE_NAME = "test-1.jpg"
IMAGE_PATH = f"{HOME}/rising-tea-images/rest-1.jpg"

if IS_COLAB:
    from google.colab import output

    output.enable_custom_widget_manager()

from jupyter_bbox_widget import BBoxWidget

widget = BBoxWidget()
widget.image = encode_image(IMAGE_PATH)
widget


# end of widget helper functions
def create_point_coordinates(bboxes):
    """Convert bounding box coordinates to point array"""
    input_point = np.zeros((0, 2))
    for point in bboxes:
        point_coord = np.array([[point["x"], point["y"]]])
        input_point = np.vstack((input_point, point_coord))
    return input_point


def find_highest_point(mask):
    """Find the point with maximum y-value in the mask

    Args:
        mask: Binary mask array
    Returns:
        tuple: (x, y) coordinates of the highest point
    """
    # Get all points where mask is True
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0:
        return None

    # Find index of maximum y-coordinate
    max_y_idx = np.argmin(y_coords)

    return np.array([[x_coords[max_y_idx], y_coords[max_y_idx]]])


def predict_masks(predictor, image_path, input_point, is_foreground=True):
    """Predict masks using SAM model"""
    # Set input label (1 for foreground, 0 for background)
    input_label = np.array([1, 0]) if is_foreground else np.array([0, 1])

    # Load and set image
    image = cv2.imread(image_path)
    predictor.set_image(image)

    # Get mask predictions
    masks, scores, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=False
    )

    return image, masks


def visualize_masks_with_highest_points(
    image, fg_masks, bg_masks, input_point, input_label
):
    """Visualize masks with their highest points"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Visualize foreground mask (green)
    fg_color = np.array([0 / 255, 255 / 255, 0 / 255, 0.6])
    for mask in fg_masks:
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * fg_color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)

        # Find and plot highest point in foreground
        fg_highest = find_highest_point(mask)
        if fg_highest is not None:
            show_points(fg_highest, np.array([1]), plt.gca())

    # Visualize background mask (red)
    bg_color = np.array([255 / 255, 0 / 255, 0 / 255, 0.6])
    for mask in bg_masks:
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * bg_color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)

        # Find and plot highest point in background
        bg_highest = find_highest_point(mask)
        if bg_highest is not None:
            show_points(bg_highest, np.array([1]), plt.gca())

    # Show original input points
    show_points(input_point, input_label, plt.gca())
    plt.axis("off")
    plt.grid(True)
    plt.show()

    return fg_highest, bg_highest


def main2(sam_model, image_path, bboxes):
    # Initialize predictor
    predictor = SamPredictor(sam_model)

    # Get input points from bounding boxes
    input_point = create_point_coordinates(bboxes)

    # Generate foreground and background masks
    image, fg_masks = predict_masks(
        predictor, image_path, input_point, is_foreground=True
    )
    _, bg_masks = predict_masks(predictor, image_path, input_point, is_foreground=False)

    # Visualize masks with highest points
    fg_highest, bg_highest = visualize_masks_with_highest_points(
        image, fg_masks, bg_masks, input_point, np.array([1, 0])
    )

    print("Tea level (y-coordinate):", fg_highest[0][1])
    print("Rim level (y-coordinate):", bg_highest[0][1])
    difference = bg_highest[0][1] - fg_highest[0][1]
    print("Distance from rim to tea:", difference, "pixels")
    return fg_highest[0][1], bg_highest[0][1]


# Usage:
# highest_fg, highest_bg = main2(sam, IMAGE_PATH, widget.bboxes)


def extract_frames(video_path, output_dir):
    """Extract frames from video at 1-second intervals and save them to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame if it corresponds to a second mark (based on FPS)
        if frame_count % int(fps) == 0:
            frame_path = os.path.join(output_dir, f"frame-{save_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            save_count += 1

        frame_count += 1

    cap.release()
    return save_count


def get_user_points(image_path):
    """Get user points using BBoxWidget"""
    widget = BBoxWidget()
    widget.image = encode_image(image_path)
    display(widget)

    return widget.bboxes


def process_points(bboxes):
    """Process bounding boxes to identify rim and tea points"""
    if len(bboxes) != 2:
        raise ValueError("Please select exactly two points")

    points = [(box["y"], box) for box in bboxes]
    points.sort(reverse=True)  # Sort by y-coordinate (highest first)

    rim_point = points[0][1]  # Point with higher y-coordinate
    tea_point = points[1][1]  # Point with lower y-coordinate

    return rim_point, tea_point


def generate_masks(predictor, image_path, rim_point, tea_point):
    """Generate masks for rim and tea"""
    # Convert points to numpy arrays
    rim_coords = np.array([[rim_point["x"], rim_point["y"]]])
    tea_coords = np.array([[tea_point["x"], tea_point["y"]]])

    # Get masks
    image = cv2.imread(image_path)
    predictor.set_image(image)

    # Generate rim mask (background)
    rim_masks, _, _ = predictor.predict(
        point_coords=rim_coords,
        point_labels=np.array([0]),  # Background
        multimask_output=False,
    )

    # Generate tea mask (foreground)
    tea_masks, _, _ = predictor.predict(
        point_coords=tea_coords,
        point_labels=np.array([1]),  # Foreground
        multimask_output=False,
    )

    return image, rim_masks, tea_masks


def visualize_masks(image, rim_masks, tea_masks):
    """Visualize masks for user approval"""
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Show rim mask in red
    if rim_masks is not None:
        show_mask(rim_masks[0], plt.gca(), color=np.array([1, 0, 0, 0.5]))

    # Show tea mask in green
    if tea_masks is not None:
        show_mask(tea_masks[0], plt.gca(), color=np.array([0, 1, 0, 0.5]))

    plt.axis("off")
    plt.show()


def process_video_frames(video_path, sam_model):
    """Main function to process video frames"""
    # Extract frames
    frames_dir = os.path.join(os.path.dirname(video_path), "frames")
    frame_count = extract_frames(video_path, frames_dir)

    # Initialize predictor
    predictor = SamPredictor(sam_model)

    # Get user input on first frame
    first_frame = os.path.join(frames_dir, "frame-0000.jpg")
    bboxes = get_user_points(first_frame)
    rim_point, tea_point = process_points(bboxes)

    # Generate and show masks for first frame
    image, rim_masks, tea_masks = generate_masks(
        predictor, first_frame, rim_point, tea_point
    )
    visualize_masks(image, rim_masks, tea_masks)

    # Get user approval
    approval = input("Do you approve these masks? (yes/no): ")
    if approval.lower() != "yes":
        return

    # Process all frames
    results = []
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f"frame-{i:04d}.jpg")
        image, rim_masks, tea_masks = generate_masks(
            predictor, frame_path, rim_point, tea_point
        )

        # Find highest points
        if rim_masks is not None and tea_masks is not None:
            rim_highest = find_highest_point(rim_masks[0])
            tea_highest = find_highest_point(tea_masks[0])

            if rim_highest is not None and tea_highest is not None:
                distance = rim_highest[0][1] - tea_highest[0][1]
                print(f"Frame {i}: Distance from rim to tea: {distance} pixels")
                results.append((i, distance))

    return results


# Example usage
video_path = "rising-tea.mp4"
results = process_video_frames(video_path, sam)
