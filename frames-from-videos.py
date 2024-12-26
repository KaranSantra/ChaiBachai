import os
import cv2
import shutil


def main():

    # Create output directory if it doesn't exist
    output_dir = "rising-tea-images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video capture
    cap = cv2.VideoCapture("rising-tea.MP4")

    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Number of frames to skip for 1-second interval
    frame_count = 0
    save_count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process every nth frame (1-second interval)
        if frame_count % frame_interval == 0:
            save_path = os.path.join(output_dir, f"rest-{save_count}.jpg")
            cv2.imwrite(save_path, frame)
            save_count += 1
        frame_count += 1
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    # Zip the output directory

    shutil.make_archive("rising-tea-images", "zip", "rising-tea-images")


if __name__ == "__main__":
    main()
