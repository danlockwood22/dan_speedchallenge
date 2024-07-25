import cv2
import torch

FPS = 20

def detect_speed(flow):
    # Calculate the average flow in the image
    avg_flow = flow.mean(axis=0).mean(axis=0)
    speed = (avg_flow[0]**2 + avg_flow[1]**2)**0.5
    return speed

# Read the video file
video_path = "data/train.mp4"
cap = cv2.VideoCapture(video_path)

output = []

prev = cap.read()[1]
prev_bw = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
while cap.isOpened():
    ret, frame = cap.read()
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_bw, frame_bw, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    speed = detect_speed(flow)
    print(speed)
    output.append(speed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()