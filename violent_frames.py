import cv2
import os

# Root path of the Violent Flows dataset
dataset_path = "violent_flows"

# Output folder for frames
output_root = "violent_flows_frames"

# Loop through folders 1 to 5
for folder_num in range(1, 6):
    for category in ["normal", "panic"]:
        folder_path = os.path.join(dataset_path, str(folder_num), category)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        # Loop through all video files in the folder
        for video_file in os.listdir(folder_path):
            if not video_file.lower().endswith((".avi", ".mp4", ".mov")):
                continue  # skip non-video files

            video_path = os.path.join(folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                continue

            # Output folder for this video
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(output_root, str(folder_num), category, video_name)
            os.makedirs(output_folder, exist_ok=True)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

            cap.release()
            print(f"Saved {frame_count} frames for {video_path}")