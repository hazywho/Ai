from ultralytics import YOLO
import cv2
import faceCutter

#setup and get source, process source so only face left
source = r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\eye-tracking-5\train\images\WIN_20221130_13_36_24_Pro_jpg.rf.0943b0639a32e323871b3ace9499f9eb.jpg"
preprocessedImage = faceCutter.cutOut(source)
model = YOLO(r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\runs\detect\train4\weights\best.pt")
prediction = model.predict(source=source)
img=cv2.imread(source)
predictedValue = prediction[0].boxes.xywh.tolist()
#iris
irisx1,irisy1,irisw1,irish1=predictedValue[0] #1
irisx2,irisy2,irisw2,irish2=predictedValue[1] #2
#eye
eyex1,eyey1,eyew1,eyeh1=predictedValue[2] #1
eyex2,eyey2,eyew2,eyeh2=predictedValue[3] #2
factor = 10
if eyex1+factor < irisx1 and eyex2+factor < irisx2:
    print("right")
if eyex1-factor > irisx1 and eyex2-factor > irisx2:
    print("left")
if eyey1+factor < irisy1 and eyey2+factor < irisy2:
    print("up")
if eyey1-factor > irisy1 and eyey2-factor > irisy2:
    print("down")

