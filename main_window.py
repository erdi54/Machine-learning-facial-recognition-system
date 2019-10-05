import sys
import os
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton
from PyQt5.QtGui import QImage,QPixmap
from PyQt5 import uic 
from PyQt5.QtCore import QTimer
from face_Card import faceCard
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.window = uic.loadUi(r"D:\ML_face\face_recognition.ui")
        self.title="Face Recognition "
        self.clicked = 0
        self.control = 0
        self.face_card= faceCard(self)
        self.face_cascade = cv2.CascadeClassifier('D:\ML_face\cascades\haarcascade_frontalface_alt.xml')
        self.face_detector = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            QMessageBox.information(self, "Error Loading cascade classifier" , "Unable to load the face	cascade classifier xml file")
            sys.exit()
        self.initUI()
        
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.timer = QTimer()
        self.window.btnStartStop.clicked.connect(lambda:self.controlTimer(1))
        #/////////////////////////////////////////////////////////////
        self.window.btnDataset.clicked.connect(lambda:self.controlTimer(2))
        #/////////////////////////////////////////////////////////////
        self.window.btnTraining.clicked.connect(self.face_training)
        #/////////////////////////////////////////////////////////////
        self.window.btnRecognition.clicked.connect(lambda:self.controlTimer(3))
        self.window.show()
    def dataParser(self):
        names = [ ]
        with open('D:\ML_face\db.csv', 'r') as data_set:
            reader = csv.reader(data_set, delimiter=',', quotechar='"')
            for data in reader:
                if data:
                   faceName = data[1]
                names.append(faceName)
            print (names)
        data_set.close() 
               
        
    def detectFaces(self):
        
        ret, frame = self.cap.read()
        scaling_factor = 0.8
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
      
        self.window.videoArea.setPixmap(QPixmap.fromImage(qImg))


    
    def face_dataset(self):
        cam = cv2.VideoCapture(0)
        with open('D:\ML_face\db.csv', 'r') as data_set:
            reader = csv.reader(data_set, delimiter=',', quotechar='"')
            for data in reader:
                if data:
                    faceId = data[0]
        face_id = faceId
        self.window.infoArea.setText("\n [INFO] Initializing face capture. Look at the camera and wait ...")
        count = 0
        while (True):
            ret, frame = cam.read()
            scaling_factor = 0.8
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite("D:/ML_face/dataset/User" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.window.videoArea.setPixmap(QPixmap.fromImage(qImg))
            
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        
            if count >= 30: # Take 30 face sample and stop video
                break
            print(count) 
        # Do a bit of cleanup
        cam.release()
        self.timer.stop()
       
        self.window.videoArea.setStyleSheet("color: rgb(0,0,255);font-weight: bold; font-size: 16pt")
        self.window.videoArea.setText(" Exiting Program and cleanup stuff")
      
        
    
    
    def face_training(self):
        path = 'dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
        # function to get the images and label data
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            faceSamples=[]
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            return faceSamples,ids
        self.window.videoArea.setText("\n  [INFO] Training faces. It will take a few seconds. Wait ... \n")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write('trainer/trainer.yml')
        self.window.videoArea.setStyleSheet("color: rgb(0,0,255);font-weight: bold; font-size: 16pt")
        self.window.videoArea.setText("{0} faces trained. Exiting Program".format(len(np.unique(ids))))
        self.window.infoArea.setText("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        self.window.btnTraining.setCheckable(False)
    
    def face_recognition(self):
            
        def print_utf8_text(image, xy, text, color):
                fontName = 'arial.ttf' 
                font = ImageFont.truetype(fontName, 24) 
                img_pil = Image.fromarray(image)  
                draw = ImageDraw.Draw(img_pil) 
                draw.text((xy[0],xy[1]), text, font=font,
                        fill=(color[0], color[1], color[2 ], 0)) 
                image = np.array(img_pil)  
                return image
        names = [ ]
        with open('D:\ML_face\db.csv', 'r') as data_set:
            reader = csv.reader(data_set, delimiter=',', quotechar='"')
            for data in reader:
                if data:
                   faceName = data[1]
                names.append(faceName)
    
        data_set.close() 
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "cascades/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)
        font = cv2.FONT_HERSHEY_SIMPLEX
        id = 0
        while True:
            ret, frame = self.cap.read()
            scaling_factor = 0.8
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                # minSize=(int(minW), int(minH)),
            )
            for (x, y, w, h) in faces:
            
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if (confidence < 100):
                   
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                   
                else:
                    id = "bilinmiyor"
                    confidence = "  {0}%".format(round(100 - confidence))
        
                color = (255,255,255)
                img=print_utf8_text(frame,(x + 5, y - 25),str(id),color)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
             
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            self.window.videoArea.setPixmap(QPixmap.fromImage(qImg))
             
    
    def Clicked(self):
       self.face_card.face_card.show()   

    def controlTimer(self,control=0):
        if control == 1:
            if not self.timer.isActive():
                self.timer.timeout.connect(self.detectFaces)
                self.cap = cv2.VideoCapture(0)  
                self.timer.start(3)                  
            else:
                self.cap.release()
                self.timer.stop()
                self.window.videoArea.setStyleSheet("color: rgb(0,0,255);font-weight: bold; font-size: 16pt")
                self.window.videoArea.setText("Camera Paused !!")
              
             
        if control == 2:
            if not self.timer.isActive():
                self.window.btnDataset.clicked.connect(self.Clicked)
                self.window.btnDataset.clicked.connect(self.dataParser)
                 
        
        if control == 3:
            if not self.timer.isActive():
                self.timer.timeout.connect(self.face_recognition)
                self.cap = cv2.VideoCapture(0)
                self.timer.start(3)
               
                    
            else:
                self.cap.release()
                self.timer.stop()
                self.window.videoArea.setStyleSheet("color: rgb(0,0,255);font-weight: bold; font-size: 16pt")
                self.window.videoArea.setText("Camera Recognition Paused !!")

                
                

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())