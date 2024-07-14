from ultralytics import YOLO
import cv2
import torch

model = YOLO(r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\rubbishPickingRobot\runs\detect\train\weights\best.pt")
cap = cv2.VideoCapture(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

while True:
    ret,frame=cap.read()
    model.predict(source=frame,show=True)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

