from ultralytics import YOLO
import cv2
import faceCutter
#import faceCutter

#setup and get source, process source so only face left
source = r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\eye-tracking-5\train\images\WIN_20221130_13_36_24_Pro_jpg.rf.0943b0639a32e323871b3ace9499f9eb.jpg"
preprocessedImage = faceCutter.cutOut(source)
model = YOLO(r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\runs\detect\train4\weights\best.pt")
prediction = model.predict(source=source)
img=cv2.imread(source)
predictedValue = prediction[0].boxes.xywh.tolist()
meanOfPredictedValue = sum([ v for v in predictedValue]) / len(predictedValue)*4

#define useful functions
def preprocess(n):
    x1 = int(n[0]+n[2]/2)
    y1 = int(n[1]+n[3]/2)
    x2 = int(n[0]-n[2]/2)
    y2 = int(n[1]-n[3]/2)
    return x1,y1,x2,y2
#joshua i need your help
def mostSimilar(n):
    #get and sort accoridng to which list is most similar
    print("peepee")

try:
    if len(predictedValue)>4:
        grouped = []
        for items in predictedValue:
            grouped.append(items)
            sorted(grouped,key=mostSimilar)

    elif len(predictedValue)==4:
        
        a1 = predictedValue[0]
        b1 = predictedValue[1]
        a2 = predictedValue[2]
        b2 = predictedValue[3]
        #small box
        irisx1,irisy1,irisw1,irish1 = preprocess(a1)
        irisx2,irisy2,irisw2,irish2 = preprocess(b1)
        img = cv2.rectangle(img,(irisx1,irisy1),(irisw1,irish1),(255,0,0),2) #plotting
        img = cv2.rectangle(img,(irisx2,irisy2),(irisw2,irish2),(255,0,0),2) #plotting
        #big box
        eyex1,eyey1,eyew1,eyeh1 = preprocess(a2)
        eyex2,eyey2,eyew2,eyeh2 = preprocess(b2)
        img = cv2.rectangle(img,(eyex1,eyey1),(eyew1,eyeh1),(0,255,0),2)
        img = cv2.rectangle(img,(eyex2,eyey2),(eyew2,eyeh2),(0,255,0),2)
    else:
        a = predictedValue[0]
        b = predictedValue[1]
        x1,y1,w1,h1 = preprocess(a)
        x2,y2,w2,h2 = preprocess(b)
        img = cv2.rectangle(img,(x1,y1),(w1,h1),(255,0,0),2)
        img = cv2.rectangle(img,(x2,y2),(w2,h2),(0,255,0),2)
    cv2.imshow("frame",img)
    cv2.waitKey()
    cv2.destroyAllWindows()
except ValueError:
    None