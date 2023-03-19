import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QLabel,
    QSlider,
    QLineEdit
)

import socket
import time
import threading
import json

sock = None
window = None
client_running = True
def client_thread_func():
    print("client_thread_func")
    #print("self = ", type(self))
    #print("sock = ", type(sock))
    count = 0
    global sock
    while client_running:
        if sock != None:
            sock.send("ACK----".encode("ascii"))
            count += 1
            #print(f"Count = %d", count)
            ack = sock.recv(7).decode("ascii")
            print(ack)
            time.sleep(1)
            
    print("Disconnected!")

import math

def deg2rad(deg):
    return ((deg*math.pi)/180.0)

def rad2deg(rad):
    return int(((rad/math.pi)*180.0))

def get_pos(sock, window):
    sock.send("GetPosL".encode("ascii"))
    left_joint_data = sock.recv(4096).decode("ascii")
    print(left_joint_data)
    left_joint_dict = json.loads(left_joint_data)
    #print("======")
    #Joint: LEFT
    window.slider_left_s0.setValue(rad2deg(left_joint_dict['left_s0']))
    window.slider_left_s1.setValue(rad2deg(left_joint_dict['left_s1']))
    window.slider_left_e0.setValue(rad2deg(left_joint_dict['left_e0']))
    window.slider_left_e1.setValue(rad2deg(left_joint_dict['left_e1']))
    window.slider_left_w0.setValue(rad2deg(left_joint_dict['left_w0']))
    window.slider_left_w1.setValue(rad2deg(left_joint_dict['left_w1']))
    window.slider_left_w2.setValue(rad2deg(left_joint_dict['left_w2']))

    #Joint: RIGHT
    sock.send("GetPosR".encode("ascii"))
    right_joint_data = sock.recv(4096).decode("ascii")
    print(right_joint_data)
    right_joint_dict = json.loads(right_joint_data)
    window.slider_right_s0.setValue(rad2deg(right_joint_dict['right_s0']))
    window.slider_right_s1.setValue(rad2deg(right_joint_dict['right_s1']))
    window.slider_right_e0.setValue(rad2deg(right_joint_dict['right_e0']))
    window.slider_right_e1.setValue(rad2deg(right_joint_dict['right_e1']))
    window.slider_right_w0.setValue(rad2deg(right_joint_dict['right_w0']))
    window.slider_right_w1.setValue(rad2deg(right_joint_dict['right_w1']))
    window.slider_right_w2.setValue(rad2deg(right_joint_dict['right_w2']))

    #print(data1)

class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Baxter Joint Move')
        self.setWindowIcon(QtGui.QIcon('baxter.png'))
        self.setGeometry(0, 0, 570, 310)

        print("width = ", self.size())

        #Values
        self.label_left_s0 = QLabel("0")
        self.label_left_s0.setGeometry(0, 0, 60, 30)
        self.label_left_s1 = QLabel("0")
        self.label_left_s1.setGeometry(0, 0, 60, 30)
        self.label_left_e0 = QLabel("0")
        self.label_left_e0.setGeometry(0, 0, 60, 30)
        self.label_left_e1 = QLabel("0")
        self.label_left_e1.setGeometry(0, 0, 60, 30)
        self.label_left_w0 = QLabel("0")
        self.label_left_w0.setGeometry(0, 0, 60, 30)
        self.label_left_w1 = QLabel("0")
        self.label_left_w1.setGeometry(0, 0, 60, 30)
        self.label_left_w2 = QLabel("0")
        self.label_left_w2.setGeometry(0, 0, 60, 30)

        self.label_right_s0 = QLabel("0")
        self.label_right_s0.setGeometry(0, 0, 60, 30)
        self.label_right_s1 = QLabel("0")
        self.label_right_s1.setGeometry(0, 0, 60, 30)
        self.label_right_e0 = QLabel("0")
        self.label_right_e0.setGeometry(0, 0, 60, 30)
        self.label_right_e1 = QLabel("0")
        self.label_right_e1.setGeometry(0, 0, 60, 30)
        self.label_right_w0 = QLabel("0")
        self.label_right_w0.setGeometry(0, 0, 60, 30)
        self.label_right_w1 = QLabel("0")
        self.label_right_w1.setGeometry(0, 0, 60, 30)
        self.label_right_w2 = QLabel("0")
        self.label_right_w2.setGeometry(0, 0, 60, 30)

        # Left Arm
        self.slider_left_s0 = QSlider(Qt.Horizontal, self)
        self.slider_left_s0.setGeometry(0, 0, 200, 30)
        self.slider_left_s0.valueChanged[int].connect(self.changeValue_left_s0)
        self.slider_left_s0.setRange(-141, 51)
        self.slider_left_s0.setValue(0)

        self.slider_left_s1 = QSlider(Qt.Horizontal, self)
        self.slider_left_s1.setGeometry(0, 0, 200, 30)
        self.slider_left_s1.valueChanged[int].connect(self.changeValue_left_s1)
        self.slider_left_s1.setRange(-123, 60)
        self.slider_left_s1.setValue(0)


        self.slider_left_e0 = QSlider(Qt.Horizontal, self)
        self.slider_left_e0.setGeometry(0, 0, 200, 30)
        self.slider_left_e0.valueChanged[int].connect(self.changeValue_left_e0)
        self.slider_left_e0.setRange(-173, 173)
        self.slider_left_e0.setValue(0)

        self.slider_left_e1 = QSlider(Qt.Horizontal, self)
        self.slider_left_e1.setGeometry(0, 0, 200, 30)
        self.slider_left_e1.valueChanged[int].connect(self.changeValue_left_e1)
        self.slider_left_e1.setRange(-3, 150)
        self.slider_left_e1.setValue(0)

        self.slider_left_w0 = QSlider(Qt.Horizontal, self)
        self.slider_left_w0.setGeometry(0, 0, 200, 30)
        self.slider_left_w0.valueChanged[int].connect(self.changeValue_left_w0)
        self.slider_left_w0.setRange(-175, 175)
        self.slider_left_w0.setValue(0)

        self.slider_left_w1 = QSlider(Qt.Horizontal, self)
        self.slider_left_w1.setGeometry(0, 0, 200, 30)
        self.slider_left_w1.valueChanged[int].connect(self.changeValue_left_w1)
        self.slider_left_w1.setRange(-90, 120)
        self.slider_left_w1.setValue(0)

        self.slider_left_w2 = QSlider(Qt.Horizontal, self)
        self.slider_left_w2.setGeometry(0, 0, 200, 30)
        self.slider_left_w2.valueChanged[int].connect(self.changeValue_left_w2)
        self.slider_left_w2.setRange(-175, 175)
        self.slider_left_w2.setValue(0)

        layout_left_s0 = QHBoxLayout()
        layout_left_s0.addWidget(QLabel("Left_S0"))
        layout_left_s0.addWidget(self.slider_left_s0)
        layout_left_s0.addWidget(self.label_left_s0)
        

        layout_left_s1 = QHBoxLayout()
        layout_left_s1.addWidget(QLabel("Left_S1"))
        layout_left_s1.addWidget(self.slider_left_s1)
        layout_left_s1.addWidget(self.label_left_s1)

        layout_left_e0 = QHBoxLayout()
        layout_left_e0.addWidget(QLabel("Left_E0"))
        layout_left_e0.addWidget(self.slider_left_e0)
        layout_left_e0.addWidget(self.label_left_e0)

        layout_left_e1 = QHBoxLayout()
        layout_left_e1.addWidget(QLabel("Left_E1"))
        layout_left_e1.addWidget(self.slider_left_e1)
        layout_left_e1.addWidget(self.label_left_e1)

        layout_left_w0 = QHBoxLayout()
        layout_left_w0.addWidget(QLabel("Left_W0"))
        layout_left_w0.addWidget(self.slider_left_w0)
        layout_left_w0.addWidget(self.label_left_w0)

        layout_left_w1 = QHBoxLayout()
        layout_left_w1.addWidget(QLabel("Left_w1"))
        layout_left_w1.addWidget(self.slider_left_w1)
        layout_left_w1.addWidget(self.label_left_w1)    

        layout_left_w2 = QHBoxLayout()
        layout_left_w2.addWidget(QLabel("Left_W2"))
        layout_left_w2.addWidget(self.slider_left_w2)
        layout_left_w2.addWidget(self.label_left_w2)    

        layout_left_arm = QVBoxLayout()
        layout_left_arm.setContentsMargins(5,5,5,5)
        layout_left_arm.addLayout(layout_left_s0)
        layout_left_arm.addLayout(layout_left_s1)
        layout_left_arm.addLayout(layout_left_e0)
        layout_left_arm.addLayout(layout_left_e1)
        layout_left_arm.addLayout(layout_left_w0)
        layout_left_arm.addLayout(layout_left_w1)
        layout_left_arm.addLayout(layout_left_w2)


        # Right Arm
        self.slider_right_s0 = QSlider(Qt.Horizontal, self)
        self.slider_right_s0.setGeometry(0, 0, 200, 30)
        self.slider_right_s0.valueChanged[int].connect(self.changeValue_right_s0)
        self.slider_right_s0.setRange(-141, 71)
        self.slider_right_s0.setValue(0)

        self.slider_right_s1 = QSlider(Qt.Horizontal, self)
        self.slider_right_s1.setGeometry(0, 0, 200, 30)
        self.slider_right_s1.valueChanged[int].connect(self.changeValue_right_s1)
        self.slider_right_s1.setRange(-123, 60)
        self.slider_right_s1.setValue(0)


        self.slider_right_e0 = QSlider(Qt.Horizontal, self)
        self.slider_right_e0.setGeometry(0, 0, 200, 30)
        self.slider_right_e0.valueChanged[int].connect(self.changeValue_right_e0)
        self.slider_right_e0.setRange(-173, 173)
        self.slider_right_e0.setValue(0)

        self.slider_right_e1 = QSlider(Qt.Horizontal, self)
        self.slider_right_e1.setGeometry(0, 0, 200, 30)
        self.slider_right_e1.valueChanged[int].connect(self.changeValue_right_e1)
        self.slider_right_e1.setRange(-3, 150)
        self.slider_right_e1.setValue(0)

        self.slider_right_w0 = QSlider(Qt.Horizontal, self)
        self.slider_right_w0.setGeometry(0, 0, 200, 30)
        self.slider_right_w0.valueChanged[int].connect(self.changeValue_right_w0)
        self.slider_right_w0.setRange(-175, 175)
        self.slider_right_w0.setValue(0)

        self.slider_right_w1 = QSlider(Qt.Horizontal, self)
        self.slider_right_w1.setGeometry(0, 0, 200, 30)
        self.slider_right_w1.valueChanged[int].connect(self.changeValue_right_w1)
        self.slider_right_w1.setRange(-90, 120)
        self.slider_right_w1.setValue(0)

        self.slider_right_w2 = QSlider(Qt.Horizontal, self)
        self.slider_right_w2.setGeometry(0, 0, 200, 30)
        self.slider_right_w2.valueChanged[int].connect(self.changeValue_right_w2)
        self.slider_right_w2.setRange(-175, 175)
        self.slider_right_w2.setValue(0)

        layout_right_s0 = QHBoxLayout()
        layout_right_s0.addWidget(QLabel("Right_S0"))
        layout_right_s0.addWidget(self.slider_right_s0)
        layout_right_s0.addWidget(self.label_right_s0)
        

        layout_right_s1 = QHBoxLayout()
        layout_right_s1.addWidget(QLabel("Right_S1"))
        layout_right_s1.addWidget(self.slider_right_s1)
        layout_right_s1.addWidget(self.label_right_s1)

        layout_right_e0 = QHBoxLayout()
        layout_right_e0.addWidget(QLabel("Right_E0"))
        layout_right_e0.addWidget(self.slider_right_e0)
        layout_right_e0.addWidget(self.label_right_e0)

        layout_right_e1 = QHBoxLayout()
        layout_right_e1.addWidget(QLabel("Right_E1"))
        layout_right_e1.addWidget(self.slider_right_e1)
        layout_right_e1.addWidget(self.label_right_e1)

        layout_right_w0 = QHBoxLayout()
        layout_right_w0.addWidget(QLabel("Right_W0"))
        layout_right_w0.addWidget(self.slider_right_w0)
        layout_right_w0.addWidget(self.label_right_w0)

        layout_right_w1 = QHBoxLayout()
        layout_right_w1.addWidget(QLabel("Right_w1"))
        layout_right_w1.addWidget(self.slider_right_w1)
        layout_right_w1.addWidget(self.label_right_w1)     

        layout_right_w2 = QHBoxLayout()
        layout_right_w2.addWidget(QLabel("Right_W2"))
        layout_right_w2.addWidget(self.slider_right_w2)
        layout_right_w2.addWidget(self.label_right_w2)    

        layout_right_arm = QVBoxLayout()
        layout_right_arm.setContentsMargins(5,5,5,5)
        layout_right_arm.addLayout(layout_right_s0)
        layout_right_arm.addLayout(layout_right_s1)
        layout_right_arm.addLayout(layout_right_e0)
        layout_right_arm.addLayout(layout_right_e1)
        layout_right_arm.addLayout(layout_right_w0)
        layout_right_arm.addLayout(layout_right_w1)
        layout_right_arm.addLayout(layout_right_w2)


        #Main window
        layout_connection = QHBoxLayout()
        #layout_connection.setGeometry(QRect(0, 0, 570, 310))
        layout_connection.addWidget(QLabel("IP Address"))
        self.text_ip = QLineEdit ("192.168.1.251")
        layout_connection.addWidget(self.text_ip)
        self.button_connect = QPushButton("Connect")
        self.button_connect.clicked.connect(self.button_connect_clicked)
        layout_connection.addWidget(self.button_connect)
        self.button_get_pos = QPushButton("Get Position")
        self.button_get_pos.clicked.connect(self.button_get_pos_clicked)
        layout_connection.addWidget(self.button_get_pos)
        self.button_set_pos = QPushButton("Set Position")
        self.button_set_pos.clicked.connect(self.button_set_pos_clicked)
        layout_connection.addWidget(self.button_set_pos)
        self.button_disconnect = QPushButton("Disconnect")
        self.button_disconnect.clicked.connect(self.button_disconnect_clicked)
        layout_connection.addWidget(self.button_disconnect)

        layout_jogging = QHBoxLayout()
        layout_jogging.addLayout(layout_left_arm)
        layout_jogging.addLayout(layout_right_arm)

        layout = QVBoxLayout()
        layout.addLayout(layout_connection)
        layout.addLayout(layout_jogging)

        self.setLayout(layout)
        
        self.button_get_pos.setEnabled(False)
        self.button_set_pos.setEnabled(False)


    def button_connect_clicked(self):
        global sock 
        global window
        ip_address = self.text_ip.text()
        print("button_connect_clicked = ", ip_address)
        sock = socket.socket()
        sock.connect((ip_address, 3000))  # connect to the server
        print("Connected!")
        get_pos(sock, window)
        window.button_connect.setEnabled(False)
        window.button_get_pos.setEnabled(True)
        window.button_set_pos.setEnabled(True)
        
        client_thread = threading.Thread(
            target=client_thread_func,
            args=()
        )
        client_thread.start()

    def button_get_pos_clicked(self):
        global sock 
        global window
        get_pos(sock, window)

    def button_set_pos_clicked(self):
        #get_pos(sock, window)
        global sock, window
        print("button_set_pos_clicked")
        left_joint_dict = {}
        left_joint_dict['left_s0'] = deg2rad(window.slider_left_s0.value())
        left_joint_dict['left_s1'] = deg2rad(window.slider_left_s1.value())
        left_joint_dict['left_e0'] = deg2rad(window.slider_left_e0.value())
        left_joint_dict['left_e1'] = deg2rad(window.slider_left_e1.value())
        left_joint_dict['left_w0'] = deg2rad(window.slider_left_w0.value())
        left_joint_dict['left_w1'] = deg2rad(window.slider_left_w1.value())
        left_joint_dict['left_w2'] = deg2rad(window.slider_left_w2.value())
        #print("left_joint_dict = ", left_joint_dict)
        sock.send("SetPosL".encode("ascii"))
        left_joint_json = json.dumps(left_joint_dict).encode("ascii")
        print("left_joint_json = ", left_joint_json)
        l = int(len(left_joint_json))
        print(" >>> left l = ", l)
        sock.send( l.to_bytes(4, 'little'))
        sock.send(left_joint_json)
        ack = sock.recv(3).decode("ascii")
        print("Set left joint = ", ack)

        right_joint_dict = {}
        right_joint_dict['right_s0'] = deg2rad(window.slider_right_s0.value())
        right_joint_dict['right_s1'] = deg2rad(window.slider_right_s1.value())
        right_joint_dict['right_e0'] = deg2rad(window.slider_right_e0.value())
        right_joint_dict['right_e1'] = deg2rad(window.slider_right_e1.value())
        right_joint_dict['right_w0'] = deg2rad(window.slider_right_w0.value())
        right_joint_dict['right_w1'] = deg2rad(window.slider_right_w1.value())
        right_joint_dict['right_w2'] = deg2rad(window.slider_right_w2.value())
        #print(right_joint_dict)
        sock.send("SetPosR".encode("ascii"))
        right_joint_json = json.dumps(right_joint_dict).encode("ascii")
        print("right_joint_json = ", right_joint_json)
        l = int(len(right_joint_json))
        print(" >>> right l = ", l)
        sock.send( l.to_bytes(4, 'little'))
        sock.send(right_joint_json)
        ack = sock.recv(3).decode("ascii")
        print("Set right joint = ", ack)
        pass

    def button_disconnect_clicked(self):
        print("button_disconnect_clicked")
        global sock, window
        if sock != None:
            sock.close()
        window.button_connect.setEnabled(True)
        window.button_get_pos.setEnabled(False)
        window.button_set_pos.setEnabled(False)
    #Left Arm
    def changeValue_left_s0(self, value):
        print("left_s0 = ", value)
        self.label_left_s0.setText(str(value))

    def changeValue_left_s1(self, value):
        print("left_s1 = ", value)
        self.label_left_s1.setText(str(value))

    def changeValue_left_e0(self, value):
        print("left_e0 = ", value)
        self.label_left_e0.setText(str(value))

    def changeValue_left_e1(self, value):
        print("left_e1 = ", value)
        self.label_left_e1.setText(str(value))

    def changeValue_left_w0(self, value):
        print("left_w0 = ", value)
        self.label_left_w0.setText(str(value))

    def changeValue_left_w1(self, value):
        print("left_w1 = ", value)
        self.label_left_w1.setText(str(value))

    def changeValue_left_w2(self, value):
        print("left_w2 = ", value)
        self.label_left_w2.setText(str(value))

    #Right Arm
    def changeValue_right_s0(self, value):
        print("right_s0 = ", value)
        self.label_right_s0.setText(str(value))

    def changeValue_right_s1(self, value):
        print("right_s1 = ", value)
        self.label_right_s1.setText(str(value))

    def changeValue_right_e0(self, value):
        print("right_e0 = ", value)
        self.label_right_e0.setText(str(value))

    def changeValue_right_e1(self, value):
        print("right_e1 = ", value)
        self.label_right_e1.setText(str(value))

    def changeValue_right_w0(self, value):
        print("right_w0 = ", value)
        self.label_right_w0.setText(str(value))

    def changeValue_right_w1(self, value):
        print("right_w1 = ", value)
        self.label_right_w1.setText(str(value))

    def changeValue_right_w2(self, value):
        print("right_w2 = ", value)
        self.label_right_w2.setText(str(value))

    def resizeEvent(self, event):
        print("Window has been resized")
        #print(event.size())
        #QtWidgets.QMainWindow.resizeEvent(self, event)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    #global window
    window = Window()
    window.show()
    sys.exit(app.exec_())