#simple face recognition program
import cv2
import face_recognition


ellon = face_recognition.load_image_file('elon-musk.jpg')#load the image
ellon = cv2.cvtColor(ellon,cv2.COLOR_BGR2RGB)#change the color space to rgb format

ellontest = face_recognition.load_image_file('elon-musktest.jpg')#load the image
ellontest = cv2.cvtColor(ellontest,cv2.COLOR_BGR2RGB)#change the color space to rgb format

ellon_loc = face_recognition.face_locations(ellon)[0]#get face location cordination
ellon_encoding = face_recognition.face_encodings(ellon)[0]#get face encoding
#draw rectangle on the face
cv2.rectangle(ellon,(ellon_loc[3],ellon_loc[0]),(ellon_loc[1],ellon_loc[2]),(255,0,255),2)

ellontest_loc = face_recognition.face_locations(ellontest)[0]#get face location cordination
ellontest_encoding = face_recognition.face_encodings(ellontest)[0]#get face encoding
#draw rectangle on the face
cv2.rectangle(ellontest,(ellontest_loc[3],ellontest_loc[0]),(ellontest_loc[1],ellontest_loc[2]),(255,0,255),2)

result = face_recognition.compare_faces([ellon_encoding],ellontest_encoding)
facedistance = face_recognition.face_distance([ellon_encoding],ellontest_encoding)
print(result,facedistance)
cv2.putText(ellontest, f'{result[0]}, {round(facedistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)
cv2.imshow('Elllon',ellon)#display the image
cv2.imshow('Ellontest',ellontest)#display the image






cv2.waitKey(0)#It helps the image to persist

cv2.destroyAllWindows()# It helps to clear the memory after close


