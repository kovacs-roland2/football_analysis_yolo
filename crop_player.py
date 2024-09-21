from utils import read_video
from trackers import Tracker
import cv2

#Read Video
video_frames = read_video('input_videos/08fd33_4.mp4')

#Initialize Tracker
tracker = Tracker('models/best.pt')

tracks = tracker.get_object_tracks(video_frames,
                                    read_from_stub=True,
                                    stub_path='stubs/track_stubs.pkl')

#Save cropped image of a player
for track_id, player in tracks['players'][0].items():
    bbox = player['bbox']
    frame = video_frames[0]

    #crop bbox from frame
    cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #save the cropped image
    cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)

    break