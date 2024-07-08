from ultralytics import YOLO
import cv2
model = YOLO(r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\EyeAssist\yolov8n.pt")
path = r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\eye-tracking-5\train\images\WIN_20221130_13_36_33_Pro_jpg.rf.25eb7232c5994972629bf98527c6e703.jpg"
image = cv2.imread(path)
def cutOut(image, log=True):
    prediction = model.predict(source=image,show=True)
    if log:
        print(prediction[0].boxes)
    try:
        value = [0,0,0,0]
        predictedValues = prediction[0].boxes.xywh.tolist()
        predictedObjects = prediction[0].boxes.cls.tolist()
        for index,item in enumerate(predictedObjects):
            if item==0:
                if predictedValues[index][2]+predictedValues[index][3] > value[2]+value[3]:
                    value = predictedValues[index]
        x,y,w,h = value
        return image[int(x-w/2):int(x+w/2), int(y-h/2):int(y+h/2)]
    except ValueError:
        return None

# cv2.imshow("frame",cutOut(image=image))
# cv2.waitKey()
# cv2.destroyAllWindows()