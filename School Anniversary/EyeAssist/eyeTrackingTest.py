from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\runs\detect\train\weights\best.pt")
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    prediction = model.predict(source=frame)
    try:
        x,y,w,h=prediction[0].boxes.xywh.tolist()
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,225,0),1)
        print(prediction[0].names[int(prediction[0].boxes.cls)])
    except ValueError or TypeError:
        None
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
cap.release()
cv2.destroyAllWindows()