from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self) -> None:
        """
        Initializes the system for managing team colors and player-team associations using KMeans clustering.

        Parameters:
        -----------
        None
            This constructor takes no parameters.

        Returns:
        --------
        None
            The constructor does not return anything. It initializes attributes related to team colors, KMeans clustering, 
            and player-team associations.
            
        Attributes:
        -----------
        team_colors : dict
            A dictionary to store the color information for different teams. Keys represent teams, and values represent
            color information.

        kmeans : KMeans or None
            A KMeans clustering model for identifying team colors from player data. Initially set to None and can be 
            initialized later when needed.

        player_team_dict : dict
            A dictionary to map players to their corresponding teams. Keys represent player identifiers, and values represent 
            the associated team.
        """
        self.team_colors = {}
        self.kmeans = None
        self.player_team_dict = {}

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """
        Applies KMeans clustering to an image to segment it into two dominant color clusters.

        Parameters:
        -----------
        image : numpy.ndarray
            The input image from which to extract color clusters, represented as a NumPy array with dimensions (height, width, 3).

        Returns:
        --------
        KMeans
            A trained KMeans model with 2 clusters, where each cluster represents one of the two dominant colors in the image.
        """
        img_2d = image.reshape(-1, 3)
        return KMeans(n_clusters=2, init="k-means++", n_init=1).fit(img_2d)
    
    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Extracts the dominant kit color of a player by analyzing the upper half of their bounding box.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) from which the player's color is to be extracted. The frame is represented as a NumPy array.
            
        bbox : list
            A list representing the bounding box coordinates of the detected player in the format [x1, y1, x2, y2],
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

        Returns:
        --------
        kit_color : numpy.ndarray
            A NumPy array representing the RGB color of the player's kit as determined by KMeans clustering.
        """
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        th_img = cropped_image[0:int(cropped_image.shape[0] / 2), :]
        kmeans = self.get_clustering_model(th_img)
        labels = kmeans.labels_
        clustered_img = labels.reshape(th_img.shape[0], th_img.shape[1])
        
        # Extract clusters from the four corners of the cropped image
        corner_clusters = [clustered_img[0, 0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, -1]]
        
        # Determine the most common cluster in the corners (assumed to be background color)
        background_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        # Assume the kit color is in the other cluster
        kit_cluster = 1 - background_cluster
        
        # Retrieve the color associated with the kit cluster
        kit_color = kmeans.cluster_centers_[kit_cluster]

        return kit_color
    
    def assign_team_color(self, frame: np.ndarray, player_detections: dict) -> None:
        """
        Assigns team colors by analyzing the colors of kit detected on players in a frame using KMeans clustering.

        Parameters:
        -----------
        frame : numpy.ndarray
            A single video frame (image) represented as a NumPy array, from which kit colors are extracted.
            
        player_detections : dict
            A dictionary of player detections. The keys represent player identifiers, and the values contain detection 
            information, including bounding boxes ('bbox') for each detected player.

        Returns:
        --------
        None
            This function does not return anything. It assigns the detected team colors to the 'team_colors' attribute.
            
        Attributes Updated:
        -------------------
        kmeans : KMeans
            Stores the KMeans clustering model fitted on the detected player colors.
            
        team_colors : dict
            A dictionary with two entries representing the team colors. Each team is assigned a cluster center from the KMeans
            clustering:
            - `team_colors[1]`: The color representing the first team.
            - `team_colors[2]`: The color representing the second team.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        """
        Determines the team ID for a player based on their color and a KMeans clustering model.

        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame (image) from which the player's color is to be extracted, represented as a NumPy array.
            
        player_bbox : list
            A list representing the bounding box coordinates of the detected player in the format [x1, y1, x2, y2],
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

        player_id : int
            A unique identifier for the player whose team is being determined.

        Returns:
        --------
        team_id : int
            The ID of the team to which the player is assigned. The ID is based on the color of the player's kit
            as determined by KMeans clustering (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Adjust to 1-based indexing

        self.player_team_dict[player_id] = team_id

        return team_id