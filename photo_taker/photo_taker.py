import cv2
import uuid
import mediapipe as mp


# Load the MediaPipe Face Detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(0.8)

padding = 20

cap = cv2.VideoCapture(0)
dataset_dir = 'dataset'
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_detection.process(frame_rgb)
    faces = []
    if results.detections:
        for detection in results.detections:
            location = detection.location_data
            bboxC = location.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # add face cropped to faces list
            face = frame[y:y+h, x:x+w]
            # resize 
            face = cv2.resize(face, (128, 128))
            faces.append(face)
            
            # add padding to the face
            x -= padding
            y -= padding
            w += padding*2
            h += padding*2
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    
    if key == 27:
        break
    elif key == ord('a'):
        for face in faces:
            cv2.imwrite(f"{dataset_dir}/alert/image_{uuid.uuid4()}.jpg", face)
        
    elif key == ord('d'):
        for face in faces:
            cv2.imwrite(f"{dataset_dir}/drowsiness/image_{uuid.uuid4()}.jpg", face)
        
    

cap.release()
cv2.destroyAllWindows()