import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
FPS = 20  # Frames per second

# Read the video file
video_path = "data/train.mp4"
cap = cv2.VideoCapture(video_path)

# Initializing the output list
output = []

# Read the first frame and resize it
ret, prev = cap.read()
    
prev_bw = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

total_displacement = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback's method
    flow = cv2.calcOpticalFlowFarneback(prev_bw, frame_bw, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Compute the average displacement
    avg_displacement = np.mean(magnitude)
    total_displacement += avg_displacement
    frame_count += 1

    # Calculate speed (meters per second)
    speed = (avg_displacement) * FPS
    output.append(speed)

    prev_bw = frame_bw

cap.release()
cv2.destroyAllWindows()

# Read the content of train.txt
with open("data/train.txt", "r") as file:
    true = file.readlines()

# Remove newline characters from each line
true = [float(line.strip()) for line in true]

# Plot the results
plt.plot(output, label='Calculated Speed')
plt.plot(true, label='Ground Truth')
plt.legend()
plt.show()
