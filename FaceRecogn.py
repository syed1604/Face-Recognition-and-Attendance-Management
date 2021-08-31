#importing libraries
import cv2
import face_recognition as fr
import os
import numpy as np
from datetime import datetime
#declaring the path
path="face_data\\"

#creating list of images and extracting names from images
images=[]
names=[]
namesList=os.listdir(path)
print(namesList)
for i in namesList:
    img=cv2.imread(f'{path}/{i}')
    images.append(img)
    names.append(os.path.splitext(i)[0])
print(names)

#encoding faces
def encodings(images):
    encoded=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=fr.face_encodings(img)[0]
        encoded.append(encode)
    return encoded

#marking attendance in a csv file
def markAttendance(name):
    with open("attendance.csv",'r+') as a:    
        myDataList = a.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            a.writelines(f'\n{name},{dtString}')
#markAttendance('abc')

#verfiying the encodings
encodingsknown= encodings(images)
print(len(encodingsknown))

#intializing camera 
capt=cv2.VideoCapture(0)

while True:
    read,img=capt.read()
    #resizing the image from webcam for faster processing
    rimg=cv2.resize(img,(0,0),None, 0.25,0.25)
    rimg=cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    
    facescurr=fr.face_locations(rimg)
    encodedfacecurr=fr.face_encodings(rimg,facescurr)
    
    #comparing the face data from webcam with trained images
    for encodeface,faceloc in zip(encodedfacecurr,facescurr):
        matches= fr.compare_faces(encodingsknown, encodeface)
        facedis=fr.face_distance(encodingsknown, encodeface)
        print(facedis)
        matchindex=np.argmin(facedis)
        
        #extracting names
        if matches[matchindex]:
            name=names[matchindex].upper()
            print(name)
        
        #creating bounding box
        y1,x2,y2,x1 = faceloc
        #resizing the image to oringinal size to have good bounding box
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
        cv2.rectangle(img,(x1,y2-30),(x2,y2),(100,100,0),cv2.FILLED)
        cv2.putText(img,name,(x1,y2),cv2.FONT_ITALIC,1,(200,200,255),2)
        markAttendance(name)
    cv2.imshow('webcam', img)
    cv2.waitKey(1)
   