from ultralytics import YOLO
import cv2
import gradio as gr
import time
import pyautogui
import torch
#pip install cv2
#pip install gradio
#pip install ultralytics

#define useful functions
def cutOut(image, log=True):
    prediction = seperator.predict(source=image,classes=[0])
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
def preprocess(n):
    x1 = int(n[0]+n[2]/2)
    y1 = int(n[1]+n[3]/2)
    x2 = int(n[0]-n[2]/2)
    y2 = int(n[1]-n[3]/2)
    return x1,y1,x2,y2
#joshua i need your help

#setup
cap = cv2.VideoCapture(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seperator = YOLO(r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\EyeAssist\faceDetection\runs\detect\train4\weights\best.pt")
model = YOLO(r"C:\Users\zanyi\OneDrive\Git hub\Ai\School Anniversary\runs\detect\train4\weights\best.pt")
model.to(device)
logging = False
sep=False

#run mainloop
def run():
    firstTime=1
    while True:
        #get source, process source so only face left
        ret, frame = cap.read()
        preprocessedImage = cutOut(frame,log=logging)
        prediction = model.predict(source=preprocessedImage if sep else frame,show_labels=False,show_conf=False,show_boxes=False)
        time.sleep(0.08)
        img = prediction[0].orig_img
        predictedValue = prediction[0].boxes.xywh.tolist()
        predictedObject = prediction[0].boxes.cls.tolist()
        print(prediction[0].boxes) if logging else None
        try:
            if predictedObject.count(0)==2 and predictedObject.count(1)==2:
                #reorder so that eye matches eye box
                a=[]
                b=[]
                e=[]
                ir=[]
                for index,value in enumerate(predictedObject):
                    if value==0:
                        a.append(predictedValue[index])
                    else:
                        b.append(predictedValue[index])
                while(len(e)!=2 and len(ir)!=2):
                    for i in a:
                        irisx = i[0]
                        irisy = i[1]
                        for j in b:
                            eyewidth=[int(j[0]-j[2]/2),int(j[0]+j[2]/2)]
                            eyeheight=[int(j[1]-j[3]/2),int(j[1]+j[3]/2)]
                            if eyewidth[0]-30 <= irisx <= eyewidth[1]+30 and eyeheight[0]-30 <= irisy <= eyeheight[1]+30:
                                e.append(j)
                                ir.append(i)
                a1 = ir[0] #xywh
                b1 = ir[1] #xywh
                a2 = e[0]#xywh
                b2 = e[1] #xywh
                #small box
                irisx1,irisy1,irisw1,irish1 = preprocess(a1)
                irisx2,irisy2,irisw2,irish2 = preprocess(b1)
                iris1xyxy=[irisx1,irisy1,irisw1,irish1]
                iris2xyxy=[irisx2,irisy2,irisw2,irish2]
                img = cv2.rectangle(img,(iris1xyxy[0],iris1xyxy[1]),(iris1xyxy[2],iris1xyxy[3]),(255,0,0),2) #plotting
                img = cv2.rectangle(img,(iris2xyxy[0],iris2xyxy[1]),(iris2xyxy[2],iris2xyxy[3]),(255,0,0),2) #plotting
                #big box
                eyex1,eyey1,eyew1,eyeh1 = preprocess(a2)
                eyex2,eyey2,eyew2,eyeh2 = preprocess(b2)
                eye1xyxy=[eyex1,eyey1,eyew1,eyeh1]
                eye2xyxy=[eyex2,eyey2,eyew2,eyeh2]
                img = cv2.rectangle(img,(eye1xyxy[0],eye1xyxy[1]),(eye1xyxy[2],eye1xyxy[3]),(0,255,0),2)
                img = cv2.rectangle(img,(eye2xyxy[0],eye2xyxy[1]),(eye2xyxy[2],eye2xyxy[3]),(0,255,0),2)
                if firstTime:
                    #calculate 3rd bounding box to determine direction
                    #eye1
                    eye1BottomLeftCorner=(eye1xyxy[0],eye1xyxy[1])
                    eye1TopRightCorner=(eye1xyxy[2],eye1xyxy[3])
                    iris1BottomLeftCorner=(iris1xyxy[0],iris1xyxy[1])
                    iris1TopRightCorner=(iris1xyxy[2],iris1xyxy[3])
                    iris1 = (a1[0],a1[1])
                    iris2 = (b1[0],b[1])
                    meanBottonLeft1=(int((eye1BottomLeftCorner[0]+iris1BottomLeftCorner[0])/2),int((eye1BottomLeftCorner[1]+iris1BottomLeftCorner[1])/2))
                    meanTopRight1=(int((eye1TopRightCorner[0]+iris1TopRightCorner[0])/2),int((eye1TopRightCorner[1]+iris1TopRightCorner[1])/2))

                    #eye2
                    eye2BottomLeftCorner=(eye2xyxy[0],eye2xyxy[1])
                    eye2TopRightCorner=(eye2xyxy[2],eye2xyxy[3])
                    iris2BottomLeftCorner=(iris2xyxy[0],iris2xyxy[1])
                    iris2TopRightCorner=(iris2xyxy[2],iris2xyxy[3])
                    meanBottonLeft2=(int((eye2BottomLeftCorner[0]+iris2BottomLeftCorner[0])/2),int((eye2BottomLeftCorner[1]+iris2BottomLeftCorner[1])/2))
                    meanTopRight2=(int((eye2TopRightCorner[0]+iris2TopRightCorner[0])/2),int((eye2TopRightCorner[1]+iris2TopRightCorner[1])/2))
                    firstTime=0
                img = cv2.rectangle(img,meanBottonLeft1,meanTopRight1,(0,0,255),1) #bounding box1
                img = cv2.rectangle(img,meanBottonLeft2, meanTopRight2,(0,0,255),1) #bounding box2
                

                #determine eye direction
                if iris1[0]<meanBottonLeft1[0] or iris2[0]<meanBottonLeft2[0]:
                    yield "Left"
                if iris1[0]>meanTopRight1[0] or iris2[0]>meanTopRight2[0]:
                    yield "Right"
                if iris1[1] > meanTopRight1[1] or iris2[1] > meanTopRight2[1]:
                    yield "Up"
                if iris1[1] < meanBottonLeft1[1] or iris2[1] < meanBottonLeft2[1]:
                    yield "Down"
                if (meanBottonLeft1[0] < iris1[0] < meanTopRight1[0] and meanBottonLeft1[1] < iris1[1] < meanTopRight1[1]) or (meanBottonLeft2[0] < iris2[0] < meanTopRight2[0] and meanBottonLeft2[1] < iris2[1] < meanTopRight2[1]):
                    yield "Middle"
            else:
                yield "no eye detected"

            cv2.imshow("frame",img)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                print("Terminating....")
                cap.release()
                cv2.destroyAllWindows()
                break
        except ValueError:
            continue
    demo.close()
    print("Terminated")
    exit()

#gradio interface
with gr.Blocks() as demo:
    submit_btn = gr.Button(value="Run")
    display = gr.Textbox()
    submit_btn.click(fn=run,outputs=display)

demo.launch()