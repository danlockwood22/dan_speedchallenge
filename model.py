import cv2
import torch
import numpy as np

FPS = 20

def normalize_flow(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = (mag - mag.min()) / (mag.max() - mag.min())
    ang = (ang - ang.min()) / (ang.max() - ang.min())
    return np.stack((mag, ang), axis=-1)

# Read the video file
video_path = "data/train.mp4"
cap = cv2.VideoCapture(video_path)

prev = cap.read()[1]
prev_bw = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

flow_data = []
while cap.isOpened():
    ret, frame = cap.read()
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_bw, frame_bw, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_norm = normalize_flow(flow)
    flow_data.append(flow_norm)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()



normalized_flows = [normalize_flow(flow) for flow in flow_data]

# Read the validation file
val_path = "train.txt"
speed_validation = []
with open(val_path, "r") as f:
    for line in f:
        speed_validation.append(float(line))

# Train the model
X = torch.tensor(normalized_flows)
y = torch.tensor(speed_validation)

import torch.nn as nn
import torch.nn.functional as F

class SpeedPredictionModel(nn.Module):
    def __init__(self):
        super(SpeedPredictionModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Assuming input size is 64x64
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SpeedPredictionModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
batch_size = 16
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
