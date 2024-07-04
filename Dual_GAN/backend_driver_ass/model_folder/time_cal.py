from moviepy.editor import VideoFileClip

def get_video_duration(file_path):
    try:
        clip = VideoFileClip(file_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    file_path = r'C:\Users\sarth\Desktop\ML\New folder\collected_data\subject4\vid.avi'
    duration = get_video_duration(file_path)
    if duration is not None:
        print(f"The duration of the video is {duration:.7f} seconds.")
