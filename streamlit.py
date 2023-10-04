import streamlit as st
import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image

# Streamlit setup
st.title("Live Webcam Feed with Face Recognition")

# Load the trained ResNet model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming 2 classes: person1 and person2
model.load_state_dict(torch.load('fine_tuned_resnet50.pth', map_location='cpu'))
model.eval()

# Define the data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize MediaPipe components
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Streamlit components
FRAME_WINDOW = st.image([])

# Capture video from webcam
camera = cv2.VideoCapture(0)

# Process each frame
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = face_detection.process(frame)

        # Draw face detections and recognition on the image
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
                
                # Get the bounding box coordinates
                boxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(boxC.xmin * iw), int(boxC.ymin * ih), int(boxC.width * iw), int(boxC.height * ih)
                
                # Extract face
                face = frame[y:y+h, x:x+w]
                pil_image = Image.fromarray(face)
                face_tensor = data_transforms(pil_image)
                face_tensor = face_tensor.unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(face_tensor)
                    _, predicted = outputs.max(1)
                    label = "person1" if predicted.item() == 0 else "person2"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        FRAME_WINDOW.image(frame)

# Release the webcam
camera.release()
