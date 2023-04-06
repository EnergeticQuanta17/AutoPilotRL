import time
import threading

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QGridLayout
from PyQt5.QtGui import QFont, QFontMetrics

from runner_trainer import *


class SenderClass(QObject):
    send_info = pyqtSignal(str, list)

    def __init__(self):
        super().__init__()
        with open("previous_request.json", "r") as f:
            self.data = json.loads(f.read())

    def send_information(self):
        for i in domain:
            to_send = domain[i]
            print("Length of list at sending side:", len(to_send))
            self.send_info.emit(i, to_send)
    
    def receive_text(self, text):
        print(f"Received text: {text}")

        self.send_info.emit(f"Received text: {text}")

class MainWindow(QWidget):
    send_text = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.all_dropdowns = []
        self.all_submits = []

        self.x_coordinate = 0
        self.y_coordinate = 0

        self.current_x = 1000
        self.current_y = 100
        self.increase_y = 200
        #increase x is dependent on the previous coming, calculate this value
        self.initUI()

    def receive_information(self, key, info):
        print("Length of list at receiving side:", len(info))
        print(max([len(i) for i in info]))

        #LABEL
        self.current_y += 100
        
        self.label = QLabel(self)
        self.label.setContentsMargins(0, 0, 0, 0)
        self.label.setText(f"{key}")
        # self.label.move(self.current_x, self.current_y)
        self.layout.addWidget(self.label, self.x_coordinate, self.y_coordinate, 1, 1)
        self.y_coordinate+=1

        # DROPDOWN
        self.dropdown = QComboBox(self)
        self.dropdown.setContentsMargins(0, 0, 0, 0)
        self.dropdown.addItems(info) # not all environments are coming
        self.current_x += 100 # calculate the length of the label

        font = QFont("Helvetica", 16)
        fm = QFontMetrics(font)

        max_length=max([fm.width(i) for i in info])
        print(max_length)
        self.dropdown.setMaximumWidth(max_length)

        self.all_dropdowns.append(self.dropdown)

        self.layout.addWidget(self.dropdown, self.x_coordinate, self.y_coordinate, 1, 1)
        self.y_coordinate-=1
        self.x_coordinate+=1

    def initUI(self):
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.setColumnStretch(1, 3)
        self.layout.setColumnStretch(2, 4)
        self.layout.setVerticalSpacing(100)

        for i in range(self.layout.rowCount()):
            self.layout.setRowStretch(i, 0.5)

        self.setWindowTitle('HyperPilotRL - Version 6')
        self.setGeometry(0, 0, 1900, 1080)
        self.setFixedSize(1900, 1080)
        self.show()

    def send_text_signal(self):
        text = self.textbox.text()
        self.send_text.emit(text)

if __name__ == "__main__":
    app = QApplication([])

    sender = SenderClass()
    receiver_window = MainWindow()

    # Connect the send_info signal from SenderClass to the receive_information slot in ReceiverWindow
    sender.send_info.connect(receiver_window.receive_information)
    receiver_window.send_text.connect(sender.receive_text)

    # Call the send_information method in SenderClass to send the information
    
    sender.send_information()   

    app.exec_()