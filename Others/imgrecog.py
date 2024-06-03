import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
fc = 0
ret = 1
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        
        
        while (fc < frameCount  and ret):
            buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
            et, buf[fc] = cap.read()
            fc += 1
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    
cv2.namedWindow('frame 10')
cv2.imshow('frame 10', buf[9])
    
cap.release()
out.release()
cv2.destroyAllWindows()




while (fc < frameCount  and ret):

