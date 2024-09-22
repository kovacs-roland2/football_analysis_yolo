from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        img_2d = image.reshape(-1, 3)
        return KMeans(n_clusters=2, init="k-means++", n_init=1).fit(img_2d)

    def get_player_color(self, frame, bbox):
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        th_img = cropped_image[0  :int(cropped_image.shape[0]/2), :]
        kmeans = self.get_clustering_model(th_img)
        labels = kmeans.labels_
        clustered_img = labels.reshape(th_img.shape[0], th_img.shape[1])
        corner_clusters = [clustered_img[0, 0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, -1]]
        background_cluster = max(set(corner_clusters), key=corner_clusters.count)
        kit_cluster = 1 - background_cluster
        kit_color = kmeans.cluster_centers_[kit_cluster]

        return kit_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id