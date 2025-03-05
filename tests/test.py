import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import time
from collections import Counter

# Define the DrowsinessCNN class
class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DrowsinessCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessCNN(num_classes=2)

checkpoint_path = r'F:\machine_learning_studying\jupyter\graduation_project\Drowsiness\drowsiness-detection\src\models\best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Data transforms for validation/testing
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),  # Resize to match model input
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

start_time = 0
last_print_time = time.time()
state_history = []
most_common_state = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading camera")
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        
        for detection in results.detections:
            # Extract face bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)

            # Ensure the bounding box is within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop the face from the frame
            face = frame[y1:y2, x1:x2]
            if face.size == 0:  # Skip if face crop is invalid
                continue

            # Preprocess the face for the model
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face_pil = Image.fromarray(face_rgb)  # Convert to PIL Image
            input_face = val_transform(face_pil).unsqueeze(0).to(device)

            # Perform model inference
            with torch.no_grad():
                outputs = model(input_face)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]

            # Add label to the frame
            label = 'Alert' if preds == 0 else 'Drowsy'
            color = (0, 255, 0) if preds == 0 else (0, 0, 255)  # Green for Alert, Red for Drowsy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bounding box

            if label == 'Drowsy':
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if start_time == 0:
                    start_time = time.time()

                if time.time() - start_time >= 5:
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            elif label == 'Alert':
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                start_time = 0

            # Print the predicted label to the kernel
            print(f"Predicted label: {label}")
            state_history.append(label)

    # Print the most voted state every 3 seconds
    current_time = time.time()
    if current_time - last_print_time >= 3:
        if state_history:
            most_common_state = Counter(state_history).most_common(1)[0][0]
            print(f"Most voted state: {most_common_state}")
            state_history = []  # Reset the history after printing
        last_print_time = current_time

    # Display the most voted state on the frame
    if most_common_state:
        cv2.putText(frame, f"Most voted state: {most_common_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop on pressing 'esc'
    key = cv2.waitKey(1)
    if key == 27:  # esc
        print("Video stopped")
        break

cap.release()
cv2.destroyAllWindows()