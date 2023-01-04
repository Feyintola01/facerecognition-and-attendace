import cv2
import face_recognition
import os
import numpy as np

path = 'ImagesAttendance'
images = []
classNames = []

mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    current_image = cv2.imread(f'{path}/{cl}')
    images.append(current_image)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList

encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None, 0.25,0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facescurFrame = face_recognition.face_locations(imgs)
    encodecurFrame = face_recognition.face_encodings(imgs,facescurFrame)

    for encodeface, faceloc in zip(encodecurFrame,facescurFrame):
        matches =  face_recognition.compare_faces(encodeListKnown,encodeface)
        facedis = face_recognition.face_distance(encodeListKnown,encodeface)
        print(facedis)

        matchIndex=  np.argmin(facedis)
        print(classNames[matchIndex])

        if matches[matchIndex]:
           name = classNames[matchIndex].upper()
           y1,x2,y2,x1 = faceloc
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1 ,(255,255,255),2)
    cv2.imshow("webcam",img)
    cv2.waitKey(1)
