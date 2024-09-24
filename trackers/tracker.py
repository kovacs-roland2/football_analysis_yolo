from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
from utils import get_center_of_bbox, get_bbox_width
import cv2

class Tracker():
    def __init__(self, model_path: str) -> None:
        """
        Initializes the object detection and tracking system by loading a YOLO model and a ByteTrack tracker.

        Parameters:
        -----------
        model_path : str
            The file path to the pre-trained YOLO model that will be used for object detection.

        Returns:
        --------
        None
            The constructor does not return anything. It initializes the model and the tracker.
            
        Attributes:
        -----------
        model : YOLO
            An instance of the YOLO model loaded from the specified path, used for detecting objects in frames.
            
        tracker : sv.ByteTrack
            An instance of the ByteTrack tracker used for tracking detected objects across frames.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_position: list) -> list:
        """
        Interpolates and fills missing ball positions across video frames.

        Parameters:
        -----------
        ball_position : list
            A list of dictionaries representing ball positions across frames. 
            Each dictionary contains frame-specific information in the format:
            [{1: {'bbox': [x1, y1, x2, y2]}}, ...].
            If the ball is not detected in a frame, the corresponding dictionary may be empty or missing.

        Returns:
        --------
        list
            A list of dictionaries with interpolated ball positions for each frame in the format:
            [{1: {'bbox': [x1, y1, x2, y2]}}, ...].
            Missing ball positions are filled based on surrounding frames.
        """
        ball_position = [x.get(1, {}).get('bbox', []) for x in ball_position]
        df_ball_positions = pd.DataFrame(ball_position, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()  # Backfill missing values for first frames

        ball_position = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_position

    def detect_frames(self, frames: list) -> list:
        """
        Detects objects in a list of video frames using a pre-trained model.

        Parameters:
        -----------
        frames : list
            A list of video frames (NumPy arrays) to be analyzed for object detection.

        Returns:
        --------
        detections : list
            A list of detection results, where each result corresponds to the detection output
            for a batch of frames. Each detection contains details like object classes, bounding boxes,
            and confidence scores.
            
        Notes:
        ------
        - The function processes the frames in batches of 20 to improve performance.
        - The model uses a confidence threshold of 0.1 for making predictions.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
        
    def get_object_tracks(self, frames: list, read_from_stub: bool = False, stub_path: str = None) -> dict:
        """
        Tracks objects such as players, referees, and the ball in a sequence of video frames.

        Parameters:
        -----------
        frames : list
            A list of video frames (NumPy arrays) to analyze for object tracking.
            
        read_from_stub : bool, optional (default=False)
            If True, reads object tracks from a saved file (stub) instead of performing detection. 
            Requires 'stub_path' to be provided.
            
        stub_path : str, optional (default=None)
            The file path to the stub (pickle file) containing precomputed tracks. If provided and the
            stub exists, object tracks will be loaded from this file instead of being computed.

        Returns:
        --------
        tracks : dict
            A dictionary containing tracked objects for each frame. The structure is:
            - "players" : A list of dictionaries where each frame contains tracked player objects with their bounding boxes.
            - "referees" : A list of dictionaries where each frame contains tracked referee objects with their bounding boxes.
            - "ball" : A list of dictionaries where each frame contains tracked ball objects with their bounding boxes.
            
            Example structure:
            ```python
            {
                "players": [{track_id: {"bbox": [x1, y1, x2, y2]}, ...}],
                "referees": [{track_id: {"bbox": [x1, y1, x2, y2]}, ...}],
                "ball": [{1: {"bbox": [x1, y1, x2, y2]}, ...}]
            }
            ```

        Notes:
        ------
        - If `read_from_stub` is set to True and `stub_path` exists, the function loads precomputed tracks from the file.
        - If `read_from_stub` is False, the function processes frames to detect and track objects.
        - The function saves the resulting tracks to the `stub_path` if provided.
        """
    
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames)
        
        tracks = {
            "players" : [], #frames[frame[0]{track_id:{"bbox":[0,0,0,0]}, track_id:{"bbox":[0,0,0,0]}, ...}, frame[1]{...}]
            "referees" : [],
            "ball" : []
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}
            
            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #convert gk to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=='goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']
                    
            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
            
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            
        return tracks
    
    def draw_ellipse(self, frame: np.ndarray, bbox: list, color: tuple, track_id: int = None) -> np.ndarray:
        """
        Draws an ellipse and an optional rectangle with a track ID on a given video frame based on a bounding box.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) where the ellipse and rectangle will be drawn, represented as a NumPy array.

        bbox : list
            A list representing the bounding box coordinates in the format [x1, y1, x2, y2],
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

        color : tuple
            A tuple representing the color of the ellipse and rectangle in BGR format (e.g., (0, 255, 0) for green).

        track_id : int, optional
            A unique identifier for the tracked object. If provided, a rectangle and text displaying the track ID will be drawn.

        Returns:
        --------
        numpy.ndarray
            The modified video frame with the drawn ellipse, rectangle, and text (if track_id is provided).
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame

    def draw_triangle(self, frame: np.ndarray, bbox: list, color: tuple) -> np.ndarray:
        """
        Draws a filled triangle and its border on a given video frame based on a bounding box.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) where the triangle will be drawn, represented as a NumPy array.

        bbox : list
            A list representing the bounding box coordinates in the format [x1, y1, x2, y2],
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

        color : tuple
            A tuple representing the fill color of the triangle in BGR format (e.g., (0, 255, 0) for green).

        Returns:
        --------
        numpy.ndarray
            The modified video frame with the drawn triangle and its border.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 10],
            [x + 10, y - 10]
        ])
        
        cv2.drawContours(  # Draw filled triangle
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED,
        )
        
        cv2.drawContours(  # Draw triangle borders
            frame,
            [triangle_points],
            0,
            (0, 0, 0),  # Black border
            2,
        )

        return frame
    
    def draw_team_ball_control(self, frame: list, frame_num: int, team_ball_control: np.ndarray) -> list:
        """
        Draws a semi-transparent rectangle and displays ball possession statistics for two teams on a video frame.

        Parameters:
        -----------
        frame : list
            A list representing the current video frame as a multi-dimensional array (e.g., numpy array) in BGR format.

        frame_num : int
            The index of the current frame being processed, used to calculate ball possession up to this frame.

        team_ball_control : np.ndarray
            An array where each element indicates which team has ball control in the corresponding frame.
            1 represents Team 1, and 2 represents Team 2.

        Returns:
        --------
        list
            The video frame with a semi-transparent overlay and text annotations displaying the ball possession statistics for each team.
        """
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate possession values
        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        team_1_count = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_count = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        all_count = team_1_count + team_2_count
        team_1_count = team_1_count / all_count
        team_2_count = team_2_count / all_count

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_count * 100: .2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_count * 100: .2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames: list, tracks: dict, team_ball_control) -> list:
        """
        Draws annotations on video frames based on detected player, referee, and ball tracks, and draws ball control panel.

        Parameters:
        -----------
        video_frames : list
            A list of video frames (NumPy arrays) to annotate with tracked object information.
            
        tracks : dict
            A dictionary containing tracking information for players, referees, and the ball. The structure is expected to be:
            {
                "players": [{track_id: {"bbox": [x1, y1, x2, y2], "team_color": (r, g, b)}, ...}],
                "referees": [{track_id: {"bbox": [x1, y1, x2, y2]}, ...}],
                "ball": [{track_id: {"bbox": [x1, y1, x2, y2]}, ...}]
            }
        
        team_ball_control : list
            A list of team ids of the team who controls the ball in the given frame.

        Returns:
        --------
        output_video_frames : list
            A list of annotated video frames with drawn ellipses for players and referees, and triangles for the ball.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))  # Default color is red
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball' ,False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 255, 255))

            # Draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))  # Yellow for referees

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))  # Green for the ball

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
                
            output_video_frames.append(frame)
            
        return output_video_frames