from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

class SenderClass(QObject):
    send_info = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def send_information(self):
        info = "Some information"
        self.send_info.emit(info)

class ReceiverWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.label = QLabel(self)
        self.label.move(50, 50)

        self.setGeometry(600, 300, 300, 200)
        self.setWindowTitle("Receiver Window")
        self.show()

    def receive_information(self, info):
        self.label.setText(info)

if __name__ == "__main__":
    app = QApplication([])

    # Create an instance of the SenderClass and ReceiverWindow
    sender = SenderClass()
    receiver_window = ReceiverWindow()

    # Connect the send_info signal from SenderClass to the receive_information slot in ReceiverWindow
    sender.send_info.connect(receiver_window.receive_information)

    # Call the send_information method in SenderClass to send the information
    sender.send_information()
    sender.send_information()

    app.exec_()
