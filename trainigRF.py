import cv2
import os
import numpy as np

dataPath = '/Users/yousufahmed/Desktop/Yousuf data scan'
peopleList = os.listdir(dataPath)
print('People List: ', peopleList)

labels = []
facesData= []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Processing, Please Hold On')

    for fileName in os.listdir(personPath):
        print('Faces: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)

    label = label + 1

#print('labels= ', labels)
#print('Number of labels 0: ', np.count_nonzero(np.array(labels)==0))
#print('Number of labels 1: ', np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.EigenFaceRecognizer_create()


#Training the program:
print('Training... ')
face_recognizer.train(facesData, np.array(labels))

#Save learning (optimizing future process)
face_recognizer.write('modeloEigenFace1.xml')

print('Model Storage Successful...')


cv2.destroyAllWindows()