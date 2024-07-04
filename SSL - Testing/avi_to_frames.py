import cv2
import os
def convert(filepath, scale):
    # Open the video file
    #print(filepath)
    video_file = os.path.join(filepath, 'vid.avi')

    if not os.path.exists(video_file):
        print("Error: .avi file not found.")
        return

    # os.chmod(filepath+'vid.avi', 0)
    cap = cv2.VideoCapture(video_file)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()
    output_dir = os.path.join(filepath, 'frames')
    os.makedirs(output_dir, exist_ok=True)


    # Initialize frame count
    frame_count = 0

    # Read the video frame by frame
    while True:
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break
        if frame.size == 0:
            #print(f"Warning: Empty frame {frame_count}, skipping.")
            frame_count -= 1
            continue

        # Save the frame as an image
        frame_path = os.path.join(output_dir, f'frame-{frame_count}.png')
        frame = cv2.resize(frame, (scale, scale))
        if not os.path.exists(frame_path):
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        #else:
            #print(f"Frame {frame_count} already exists, skipping.")

    # Release the video capture object and close the video file
    cap.release()

    #print(f"Frames saved: {frame_count}")
