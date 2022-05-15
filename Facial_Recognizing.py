import cv2
import os

dataPath = '/Users/yousufahmed/Desktop/Yousuf data scan '
imagePaths = os.listdir(dataPath)
print('imagePaths= ', imagePaths)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Reading the storage model:
face_recognizer.read('modeloEigenFace.xml1')
#uso de camara para analizar
cap = cv2.VideoCapture(0)
#seleccona el documento a analizar:
#cap = cv2.VideoCapture('/home/franddy/Desktop/reconocimiento facial comapardor/material de prueba/faury_video_prueba.mp4')
#cap = cv2.VideoCapture('/home/franddy/Desktop/reconocimiento facial comapardor/material de prueba/InShot_20220508_054815392.mp4')
#cap = cv2.VideoCapture('/home/franddy/Desktop/reconocimiento facial comapardor/material de prueba/InShot_20220513_124111477.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3,5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x,y-5), 1, 1.3, (255, 255, 0),1, cv2.LINE_AA)

        # EigenFaces (recognizing and comparing)
        #en if se introduce el valor de tolerancia de la comparacion, mayor numero mayor tolerancia y menos presicion
        if result [1] < 5400:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Unknown. Please Send Your Information To Register', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()