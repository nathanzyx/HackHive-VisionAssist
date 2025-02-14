#
#  Usage if this file includes optional Nvidia Cuda Core usage
#
#

import cv2
import torch

import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from gtts import gTTS

import pyttsx3

cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)  # Enable OpenCL (for NVIDIA/AMD GPUs)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should return your GPU name

engine = pyttsx3.init()  # Initialize eSpeak engine

# Initialize the YOLOv8 model (you can choose a larger model if desired)
model = YOLO('yolo11l.pt')  # or 'yolov8s.pt' for better accuracy
print(f"Using device: {model.device}")


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# stream_url = "http://10.190.32.130:4747"
# cap = cv2.VideoCapture(stream_url)


with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1 # 1 for GPU, 0 for CPU
) as hands:
    frame_counter = 0
    
    lastObj = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w, _ = frame.shape

        # Process the frame with MediaPipe Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx == 1:
                    break
                # Draw hand landmarks on the frame
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist and index finger tip landmarks
                # wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert normalized coordinates to pixel coordinates
                index_base_px = np.array([int(index_base.x * w), int(index_base.y * h)])
                index_tip_px = np.array([int(index_tip.x * w), int(index_tip.y * h)])

                # Calculate pointing vector ----
                pointing_vector = index_tip_px - index_base_px

                distance = np.linalg.norm(index_tip_px - index_base_px)
                # print(f"Distance: {distance}")
                

                pointing_vector_normalized = pointing_vector / np.linalg.norm(pointing_vector)

                # # For moving ROI center in front of finger tip
                roi_displacement = 100
                roi_center = index_tip_px + (pointing_vector_normalized * roi_displacement)
                
                # # -------------------
                roi_size = 300  # Smaller size for accuracy
                x1 = int(max(roi_center[0] - roi_size // 2, 0))
                y1 = int(max(roi_center[1] - roi_size // 2, 0))
                x2 = int(min(roi_center[0] + roi_size // 2, w))
                y2 = int(min(roi_center[1] + roi_size // 2, h))

                # Extract the fingertip region
                roi = frame[y1:y2, x1:x2]

                # Check if ROI is empty
                if roi.size == 0:
                    continue  # Skip if ROI is empty
                
                
                
                # Draw the ROI rectangle on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                                
                if frame_counter % 5 == 0 or 1 == 1:
                # Perform object detection on the ROI
                    results_yolo = model.predict(source=roi, conf=0.5, imgsz=1088, verbose=False, device=device)
                    detections = results_yolo[0]

                    if len(detections.boxes) > 0:
                        
                        for box_idx, detection_box in enumerate(detections.boxes.xyxy):
                            print(f"Detected {int(detections.boxes.cls[box_idx])}")
                            if int(detections.boxes.cls[box_idx]) == 0:
                                break
                            class_id = int(detections.boxes.cls[box_idx])
                            confidence = detections.boxes.conf[box_idx]

                            
                            
                            
                        # Get the detection with the highest confidence
                        max_conf_idx = np.argmax(detections.boxes.conf.cpu().numpy())
                        detection_box = detections.boxes.xyxy[max_conf_idx].cpu().numpy().astype(int)
                        class_id = int(detections.boxes.cls[max_conf_idx])
                        detected_label = model.names[class_id]
                        
                        if lastObj != detected_label:
                            engine.say(detected_label)
                            engine.runAndWait()
                            lastObj = detected_label

                        # Coordinates relative to ROI; adjust to frame coordinates
                        dx1, dy1, dx2, dy2 = detection_box
                        dx1 += x1
                        dy1 += y1
                        dx2 += x1
                        dy2 += y1

                        # Update ROI to the new dynamic ROI based on detection
                        x1_new = max(dx1 - roi_size // 4, 0)
                        y1_new = max(dy1 - roi_size // 4, 0)
                        x2_new = min(dx2 + roi_size // 4, w)
                        y2_new = min(dy2 + roi_size // 4, h)

                        # Draw the adjusted ROI rectangle
                        cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (255, 0, 0), 2)

                        # Draw the detection bounding box
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        cv2.putText(frame, detected_label, (dx1, dy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # If no detection, you can choose to expand the ROI or handle it differently
                        pass
                
                frame_counter += 1

                # Visualize the pointing vector
                cv2.arrowedLine(frame, (index_base_px[0], index_base_px[1]), (index_tip_px[0], index_tip_px[1]),
                                (0, 0, 255), 2)
        else:
            detected_label = "No hand detected"
            if lastObj != detected_label:
                engine.say(detected_label)
                engine.runAndWait()
                lastObj = detected_label

        cv2.imshow("Dynamic ROI Object Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
