import os
import ffmpeg

def split_video(video_file, start_time, end_time):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    if os.path.exists(output_file):
        print(f"Video file {output_file} already exists.")
        return output_file

    try:
        (
            ffmpeg
            .input(video_file, ss=int(start_time))
            .output(output_file, t=(int(end_time) - int(start_time)), vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode('utf-8')}")
        
    print(f"Video: {output_file} splitting completed.")
    return output_file
    