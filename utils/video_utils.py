import cv2

def read_video(video_path: str) -> list:
    """
    Reads a video from the specified file path and extracts all frames.

    Parameters:
    -----------
    video_path : str
        The file path to the video that needs to be read.

    Returns:
    --------
    frames : list of numpy.ndarray
        A list of frames extracted from the video, where each frame is represented
        as a NumPy array containing the pixel data.
    """
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        is_frame, frame = capture.read()
        if not is_frame:
            break
        frames.append(frame) 
    return frames

def save_video(output_video_frames: list, output_video_path: str) -> None:
    """
    Saves a list of video frames to a video file at the specified path.

    Parameters:
    -----------
    output_video_frames : list of numpy.ndarray
        A list of frames, where each frame is represented as a NumPy array (typically extracted from a video).
        
    output_video_path : str
        The file path where the output video should be saved.

    Returns:
    --------
    None
        This function does not return anything. It writes the video to the specified file path.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps=24,
                          frameSize=(output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
