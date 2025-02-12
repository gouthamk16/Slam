import torch
from ultralytics import YOLO
import cv2
import numpy as np


## Functiont to extract the important features in the image using cv2.goodFeaturesToTrack
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 0, 0.01, 1)
    corners = np.intp(corners)
    edges = cv2.Canny(gray, 100, 200)
    return corners, edges

def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(f1, f2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

class VehicleTracker:
    def __init__(self, confidence_threshold=0.4):
        # Initialize YOLO model with GPU support
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # self.notifier.speak("Detection Initiated")
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # or 'yolov8n.pt' for less accuracy but faster inference
        self.model.to(self.device)

        self.frames = [np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)]
        
        # Tracking parameters
        self.confidence_threshold = confidence_threshold
        
        # Valid vehicle classes in YOLO v8
        self.vehicle_classes = [0, 1, 2, 3, 5, 7]  # car, bus, truck in YOLOv8

        self.classMap = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 100: "NA"}
        
    def process_frame(self, frame, target_fps=10):
        self.frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        corners1, edges1 = extract_features(frame)
        corners2, edges2 = extract_features(self.frames[-2])
        black_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # Get the matches between the current frame and the previous frame
        # if len(self.frames) > 1:
        #     prev_frame = self.frames[-2]
        #     matches = match_frames(corners1, corners2)
        #     for match in matches:
        #         x1, y1 = corners1[match.queryIdx].ravel()
        #         x2, y2 = corners2[match.trainIdx].ravel()
        #         cv2.line(black_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)


        for feature in corners1:
            fx, fy = feature.ravel()
            cv2.circle(black_frame, (fx, fy), 0, (0, 255, 0), 1)
        
        edges_colored = cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR)
        black_frame = cv2.addWeighted(black_frame, 0.8, edges_colored, 0.2, 0)
        
        results = self.model(frame_rgb, verbose=False)
    
        current_vehicles = []
        cls = None
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    current_vehicles.append((int(x1), int(y1), int(w), int(h)))

        if cls not in self.vehicle_classes:
            cls = 100

        for vehicle in current_vehicles:
            x, y, w, h = vehicle
            # cv2.circle(frame, (x+(w//2), y+(h//2)), 0, (0, 255, 0), 5)
            cv2.circle(black_frame, (x+(w//2), y+(h//2)), 0, (255, 0, 255), 5)
            # cv2.rectangle(black_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            vehicleClass = self.classMap[cls]
            # cv2.putText(black_frame, f"{vehicleClass}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        print("No of corner features detected: ", len(corners1)) 
        print()       
        return frame, black_frame