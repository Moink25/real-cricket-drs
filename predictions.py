import numpy as np

class BallPredictor:
    def __init__(self, ball_trajectory):
        self.ball_trajectory = ball_trajectory

    def predict_future_path(self, num_future_frames, scaling_factor):
        future_path = []
        if len(self.ball_trajectory) > 1:
            speeds = self.calculate_speeds()
            avg_speed = np.mean(speeds)

            # Adjust number of future frames based on speed
            num_future_frames = self.adjust_future_frames_based_on_speed(avg_speed, num_future_frames)

            avg_delta_x, avg_delta_y = self.calculate_average_deltas(num_future_frames)

            # Calculate the future trajectory points
            x_last, y_last = self.ball_trajectory[-1]
            for i in range(1, num_future_frames + 1):
                future_x = int(x_last + i * avg_delta_x)
                future_y = int(y_last + i * avg_delta_y)
                future_path.append((future_x, future_y))

        return future_path

    def calculate_speeds(self):
        speeds = []
        for i in range(1, len(self.ball_trajectory)):
            delta_x = self.ball_trajectory[i][0] - self.ball_trajectory[i - 1][0]
            delta_y = self.ball_trajectory[i][1] - self.ball_trajectory[i - 1][1]
            speed = np.sqrt(delta_x ** 2 + delta_y ** 2)  # Euclidean distance
            speeds.append(speed)
        return speeds

    def adjust_future_frames_based_on_speed(self, avg_speed, num_future_frames):
        if avg_speed < 10:  # Adjust threshold based on your requirements
            return 15  # More frames for slow speeds
        elif avg_speed < 20:
            return 10  # Medium frames for medium speeds
        else:
            return 5  # Less frames for fast speeds

    def calculate_average_deltas(self, num_points_for_prediction):
        total_delta_x = 0
        total_delta_y = 0
        count = 0

        for i in range(-num_points_for_prediction, -1):
            delta_x = self.ball_trajectory[i + 1][0] - self.ball_trajectory[i][0]
            delta_y = self.ball_trajectory[i + 1][1] - self.ball_trajectory[i][1]
            total_delta_x += delta_x
            total_delta_y += delta_y
            count += 1

        avg_delta_x = (total_delta_x / count)
        avg_delta_y = (total_delta_y / count)
        return avg_delta_x, avg_delta_y

    def predict_hitting_stumps(self, future_path, stump_properties):
        predicted_hitting_stumps = False
        hitting_count = 0

        stump_width, _, stump_x, stump_y = stump_properties
        for point in future_path:
            if stump_x <= point[0] <= stump_x + stump_width and point[1] >= stump_y:
                hitting_count += 1

        # Determine if hitting based on the count of future points
        hitting_probability = hitting_count / len(future_path)
        if hitting_probability > 0.2:  # More than 20% of points hitting
            predicted_hitting_stumps = True

        return predicted_hitting_stumps
