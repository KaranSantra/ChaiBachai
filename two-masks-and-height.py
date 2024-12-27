import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
import base64
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Constants
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
IS_COLAB = True
IMAGE_NAME = "test-1.jpg"
IMAGE_PATH = f"{HOME}/rising-tea-images/rest-1.jpg"


# Visualization Functions
def show_mask(mask, ax, random_color=False):
    color = (
        np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        if random_color
        else np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    )
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


# Image Processing Functions
def encode_image(filepath):
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


def find_highest_point(mask):
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0:
        return None
    max_y_idx = np.argmin(y_coords)
    return np.array([[x_coords[max_y_idx], y_coords[max_y_idx]]])


def predict_masks(predictor, image_path, input_point, is_foreground=True):
    input_label = np.array([1, 0]) if is_foreground else np.array([0, 1])
    image = cv2.imread(image_path)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=False
    )
    return image, masks


def visualize_masks_with_highest_points(
    image, fg_masks, bg_masks, input_point, input_label
):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Process foreground masks
    fg_color = np.array([0 / 255, 255 / 255, 0 / 255, 0.6])
    fg_highest = None
    for mask in fg_masks:
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * fg_color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)
        fg_highest = find_highest_point(mask)
        if fg_highest is not None:
            show_points(fg_highest, np.array([1]), plt.gca())

    # Process background masks
    bg_color = np.array([255 / 255, 0 / 255, 0 / 255, 0.6])
    bg_highest = None
    for mask in bg_masks:
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * bg_color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)
        bg_highest = find_highest_point(mask)
        if bg_highest is not None:
            show_points(bg_highest, np.array([1]), plt.gca())

    show_points(input_point, input_label, plt.gca())
    plt.axis("off")
    plt.grid(True)
    plt.show()

    return fg_highest, bg_highest


def process_tea_levels(sam_model, image_path, points):
    predictor = SamPredictor(sam_model)

    input_point = np.zeros((0, 2))
    for point in points:
        point_coord = np.array([[point["x"], point["y"]]])
        input_point = np.vstack((input_point, point_coord))
    # Generate masks
    image, fg_masks = predict_masks(
        predictor, image_path, input_point, is_foreground=True
    )
    _, bg_masks = predict_masks(predictor, image_path, input_point, is_foreground=False)

    # Visualize and get measurements
    fg_highest, bg_highest = visualize_masks_with_highest_points(
        image, fg_masks, bg_masks, input_point, np.array([1, 0])
    )

    # Calculate measurements
    tea_level = fg_highest[0][1]
    rim_level = bg_highest[0][1]
    difference = rim_level - tea_level

    print(f"Tea level (y-coordinate): {tea_level}")
    print(f"Rim level (y-coordinate): {rim_level}")
    print(f"Distance from rim to tea: {difference} pixels")

    return tea_level, rim_level


# Main execution
if __name__ == "__main__":
    # Initialize SAM model
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    # Setup widget if in Colab
    if IS_COLAB:
        from google.colab import output

        output.enable_custom_widget_manager()

    from jupyter_bbox_widget import BBoxWidget

    widget = BBoxWidget()
    widget.image = encode_image(IMAGE_PATH)

    # Process tea levels
    highest_fg, highest_bg = process_tea_levels(sam, IMAGE_PATH, widget.bboxes)
