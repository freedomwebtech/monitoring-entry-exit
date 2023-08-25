import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture(r'C:\Users\freed\ytfinalvideos\gate.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0


area1=[(551,177),(534,185),(586,217),(619,186)]
area2=[(494,200),(472,204),(529,235),(565,228)]

tracker=Tracker()
going_in={}
counter1=[]
going_out={}
counter2=[]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1120,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)

    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),2,2)
        
         

            
    
 #   cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
 #   cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
   

#    print(len(counter1))

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

