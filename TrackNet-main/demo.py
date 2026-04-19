import numpy as np
import cv2
import torch
import os
import math
from scipy.stats import multivariate_normal
import pandas as pd
from tqdm import tqdm
from collections import deque  
from models.TrackNet_Beta import TrackNetBeta


class TrackNetPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrackNetBeta().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint['model'])
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_frames(self, frames):
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (512, 288))
            frame = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
            processed.append(frame)
        return torch.cat(processed, dim=0).unsqueeze(0).to(self.device)

    def predict(self, frames):
        input_tensor = self.preprocess_frames(frames) 
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.squeeze(0).cpu().numpy()

    def detect_ball(self, heatmap, threshold=0.5):
        if heatmap.max() < threshold:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        print('The current coordinate is:({}, {})'.format(max_pos[1], max_pos[0]))  # 105,466
        return (max_pos[1], max_pos[0])

    def to_img(self, image):
        image = image * 255
        image = image.astype('uint8')
        return image

    def predict_location(self, heatmap):
        if np.amax(heatmap) == 0:

            return 0, 0, 0, 0
        else:
            
            (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]

            
            max_area_idx = 0
            max_area = rects[0][2] * rects[0][3]
            for i in range(1, len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            x, y, w, h = rects[max_area_idx]

            return x, y, w, h


    def confusion_matrix_gt(self, y_pred=None, y_true=None, tolerance=4.):

        TP = TN = FP1 = FP2 = FN = 0


        assert y_true is not None and y_pred is not None, 'Invalid input'
        y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        h_pred = y_pred > 0.5  

        if y_true is not None and y_pred is not None:
            # Predict from heatmap
            y_t = y_true
            h_p = h_pred
            bbox_true = self.predict_location(y_t)
            cx_true, cy_true = int(bbox_true[0] + bbox_true[2] / 2), int(bbox_true[1] + bbox_true[3] / 2)
            bbox_pred = self.predict_location(self.to_img(h_p))
            cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)

            if np.amax(h_p) == 0 and np.amax(y_t) == 0:
                # True Negative: prediction is no ball, and ground truth is no ball
                TN += 1
            elif np.amax(h_p) > 0 and np.amax(y_t) == 0:
                # False Positive 2: prediction is ball existing, but ground truth is no ball
                FP2 += 1
            elif np.amax(h_p) == 0 and np.amax(y_t) > 0:
                # False Negative: prediction is no ball, but ground truth is ball existing
                FN += 1
            elif np.amax(h_p) > 0 and np.amax(y_t) > 0:
                # Both prediction and ground truth are ball existing
                # Find center coordinate of the contour with max area as prediction
                dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                if dist > tolerance:
                    # False Positive 1: prediction is ball existing, but is too far from ground truth
                    FP1 += 1
                else:
                    # True Positive
                    TP += 1
            else:
                raise ValueError('Invalid input')
        else:
            raise ValueError('Invalid input')

        return (cx_pred, cy_pred), (TP, TN, FP1, FP2, FN)


    def confusion_matrix(self, y_pred=None, cx_true=None, cy_true=None, tolerance=4.):
        TP = TN = FP1 = FP2 = FN = 0
        assert y_pred is not None, 'Invalid input'
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        h_pred = y_pred > 0.5  
        h_p = h_pred.copy()
        if y_pred is not None:
            # Predict from heatmap
            bbox_pred = self.predict_location(self.to_img(h_p))
            cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
        return (cx_pred, cy_pred)


    def generate_heatmap(self, center_x, center_y, width=512, height=288, sigma=2.5):
        """Generate 2D Gaussian heatmap centered at specified coordinates."""
        x_coords = np.arange(0, width)
        y_coords = np.arange(0, height)
        mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
        coordinates = np.dstack((mesh_x, mesh_y))

        gaussian_mean = [center_x, center_y]
        covariance_matrix = [[sigma ** 2, 0], [0, sigma ** 2]]

        distribution = multivariate_normal(gaussian_mean, covariance_matrix)
        heatmap = distribution.pdf(coordinates)

        # Normalize to 0-255 range
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)

        return heatmap_uint8


    def detect_ball_prec(self, heatmap, threshold=0.5, csv_path=None):
        
        if csv_path is not None:
            
            groundtruth = self.generate_heatmap(csv_path[0], csv_path[1])
            (cx_pred, cy_pred), (tp, tn, fp1, fp2, fn) = self.confusion_matrix_gt(heatmap, groundtruth, tolerance=4)
            print('The current coordinate is:({}, {})'.format(cx_pred, cy_pred))
            return (cx_pred, cy_pred), (tp, tn, fp1, fp2, fn)
        else:
            if heatmap.max() < threshold:
                return None
           
            (cx_pred, cy_pred) = self.confusion_matrix(heatmap)
            print('The current coordinate is:({}, {})'.format(cx_pred, cy_pred))
            return (cx_pred, cy_pred) 


class VideoProcessor:
    
    def __init__(self, model_path, dot_size=3, trajectory_len=3):
        self.predictor = TrackNetPredictor(model_path)
        self.dot_size = dot_size  
        self.trajectory_len = trajectory_len 

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        with tqdm(total=total_frames, desc="�� Extracting frames", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                pbar.update(1)

        cap.release()
        print(f"✅ Extraction complete: {len(frames)} frames, {width}x{height}, {fps:.1f}FPS")
        return frames, (width, height), fps

    def group_frames(self, frames):
        groups = []
        for i in range(0, len(frames) - 2, 3):
            if i + 3 <= len(frames):
                groups.append(frames[i:i + 3])

        discarded = len(frames) - len(groups) * 3
        print(f"�� Grouping complete: {len(groups)} groups (3 frames/group), {discarded} frames discarded")
        return groups


    def group_coords(self, path):
        groups = []
        annotations_df = pd.read_csv(path)
        
        annotation_lookup = {}
        for _, row in annotations_df.iterrows():
            annotation_lookup[row['Frame']] = row

        for i in range(0, len(annotation_lookup) - 2, 3):
            if i + 3 <= len(annotation_lookup):
                groups.append([
                    (annotation_lookup[i]['X'], annotation_lookup[i]['Y']),
                    (annotation_lookup[i + 1]['X'], annotation_lookup[i + 1]['Y']),
                    (annotation_lookup[i + 2]['X'], annotation_lookup[i + 2]['Y'])
                ])

        discarded = len(annotation_lookup) - len(groups) * 3
        print(f"�� Grouping complete: {len(groups)} groups (3 frames/group), {discarded} frames discarded")
        return groups


    def scale_coordinates(self, coords, original_size):
        if coords is None:
            return None
        x, y = coords
        scale_x = original_size[0] / 512
        scale_y = original_size[1] / 288
        return (int(x * scale_x), int(y * scale_y))


    def rescale_coordinates(self, coords, original_size):
        if coords is None:
            return None
        x, y = coords
        scale_x = original_size[0] / 512
        scale_y = original_size[1] / 288
        return (int(x / scale_x), int(y / scale_y))  


    def get_metric(self, TP, TN, FP1, FP2, FN):

        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN) if (TP + TN + FP1 + FP2 + FN) > 0 else 0
        precision = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0

        return accuracy, precision, recall, f1, miss_rate

    
    def draw_ball(self, frame, ball_pos):

        if ball_pos is None:
            return frame

        
        if isinstance(ball_pos, (deque, list, tuple)) and len(ball_pos) > 0 and isinstance(ball_pos[0],
                                            (tuple, type(None))):
            points = list(ball_pos)  
        else:
            points = [ball_pos]  

        if len(points) == 0:
            return frame

       
        main_color = (0, 165, 255)  
        halo_color = (255, 255, 255)  
        rim_color = (0, 0, 0)  
        center_color = (255, 255, 255)  


        r = int(self.dot_size)
        halo_r = r + 3
        rim_r = r + 2
        center_r = max(2, r // 2)


        for p in points:
            if p is None:
                continue
            x, y = int(p[0]), int(p[1])

            cv2.circle(frame, (x, y), halo_r, halo_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), rim_r, rim_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), r, main_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), center_r, center_color, -1, lineType=cv2.LINE_AA)

        return frame


    def process_video(self, video_path, output_path="processed_video.mp4"):
        print("�� Starting shuttlecock detection...")
        print(f"�� Red dot size: {self.dot_size} pixels")
        print(f"�� Trajectory length: {self.trajectory_len} frames")  

        frames, original_size, fps = self.extract_frames(video_path)
        frame_groups = self.group_frames(frames)

        processed_frames = []
        TP = TN = FP1 = FP2 = FN = 0
        ball_detected_count = 0
        total_processed_frames = 0

        trajectory_history = deque(maxlen=self.trajectory_len)
        with tqdm(total=len(frame_groups), desc="�� Detecting shuttlecock", unit="groups") as pbar:
        
            for group in frame_groups:
                heatmaps = self.predictor.predict(group)
                for frame, heatmap in zip(group, heatmaps):

                    ball_pos_model = self.predictor.detect_ball_prec(heatmap)  
                    ball_pos_original = self.scale_coordinates(ball_pos_model, original_size)

                    trajectory_history.append(ball_pos_original) 
                    processed_frame = self.draw_ball(frame.copy(), trajectory_history) 

                    processed_frames.append(processed_frame)

                    total_processed_frames += 1
                    if ball_pos_original:
                        ball_detected_count += 1

                pbar.update(1)

        # detection_rate = (ball_detected_count / total_processed_frames) * 100
        # print(f"�� Detection stats: {ball_detected_count}/{total_processed_frames} frames ({detection_rate:.1f}%)")
        # accuracy, precision, recall, f1, miss_rate = self.get_metric(TP, TN, FP1, FP2, FN)
        # print(f"�� Accuracy:{accuracy:.3f}% Precision:{detection_rate:.3f}% Recall::{recall:.3f}% "
        #       f"F1::{f1:.3f}% Miss_Rate::{miss_rate:.3f}%")
        self.save_video(processed_frames, output_path, fps, original_size)
        return output_path


    def save_video(self, frames, output_path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        if not out.isOpened():
            raise RuntimeError(f"Cannot create video file: {output_path}")

        with tqdm(total=len(frames), desc="�� Saving video", unit="frames") as pbar:
            for frame in frames:
                out.write(frame)
                pbar.update(1)

        out.release()
        print(f"✅ Video saved successfully: {output_path}")




def main():
    model_path = "/home/.../UniTrack-main/models/TrackNetBeta.pt"
    input_video =  "/home/.../UniTrack-main/prediction/demo_video.mp4"
    output_video = "/home/.../UniTrack-main/prediction/predicted.mp4"


    RED_DOT_SIZE = 6 
    TRAJECTORY_LEN = 8

    print("=" * 60)
    print("�� Badminton Shuttlecock Detection & Tracking System")
    print("=" * 60)
    print(f"�� Model file: {model_path}")
    print(f"�� Input video: {input_video}")
    print(f"�� Output video: {output_video}")
    print(f"�� Red dot size: {RED_DOT_SIZE} pixels")
    print(f"�� Trajectory Length: {TRAJECTORY_LEN} frames")
    print("-" * 60)

    if not os.path.exists(input_video):
        print(f"❌ Error: Input video file not found: {input_video}")
        return

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return

    try:

        processor = VideoProcessor(model_path, dot_size=RED_DOT_SIZE, trajectory_len=TRAJECTORY_LEN)
        output_path = processor.process_video(input_video, output_video)
        print("=" * 60)
        print(f"�� Processing complete! Output video: {output_path}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")


if __name__ == "__main__":
    main()