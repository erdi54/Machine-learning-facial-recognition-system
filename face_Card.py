from PyQt5 import uic,QtGui,QtCore
from PyQt5.QtWidgets import QApplication,QWidget,QLineEdit,QPushButton,QMessageBox
import csv,os


class faceCard(QWidget):
    def __init__(self,parent=None):
        super(faceCard, self).__init__(parent)
        self.face_card = uic.loadUi(r"D:\ML_face\face_card.ui")
        self.parent = parent
        self.click=0
        self.initUI()

    def initUI(self):
       
        self.face_card.btnKaydet.clicked.connect(self.DataRegister)
        self.face_card.btnClose.clicked.connect(self.faceDataset_redirect )
    
    def openFile(self, address=r"D:\ML_face\db.csv"):
        if os.path.exists(address):
            kip = "r+"
        else:
            kip = "w+"
        
        return open(address,kip,encoding='utf-8')

      
    def DataRegister(self):
        self.parent.clicked += 1
        if self.parent.clicked <= 1:
            File = self.openFile()
            face_id = self.face_card.txtFace_id.text()
            name = self.face_card.txtAdiSoyadi.text()
            registry="{},{}\n".format(face_id,name)
            temp=File.readlines()
            temp.append(registry)
            File.seek(0)
            File.truncate()
            File.writelines(temp)
            File.close()
        else:
            QMessageBox.warning(self, 'Message', "More than one click")

         
    def faceDataset_redirect(self):
        self.face_card.close()
        self.parent.timer.timeout.connect(self.parent.face_dataset)
        self.parent.timer.start(3)
    