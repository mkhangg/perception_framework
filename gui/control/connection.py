
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
    QLineEdit,
    QGroupBox
)

from control.const import *

def init_connection(self):
    #Connection
    layout_connection = QHBoxLayout()
    layout_connection.addWidget(QLabel("IP Address"))
    self.text_ip = QLineEdit ("192.168.1.249")
    layout_connection.addWidget(self.text_ip)
    self.button_connect = QPushButton("Connect")
    self.button_connect.clicked.connect(self.button_connect_clicked)
    layout_connection.addWidget(self.button_connect)
    # self.button_get_pos = QPushButton("Get Position")
    # self.button_get_pos.clicked.connect(self.button_get_pos_clicked)
    # layout_connection.addWidget(self.button_get_pos)
    # self.button_set_pos_left = QPushButton("Set Left Arm")
    # self.button_set_pos_left.clicked.connect(self.button_set_pos_left_clicked)
    # layout_connection.addWidget(self.button_set_pos_left)
    # self.button_set_pos_right = QPushButton("Set Right Arm")
    # self.button_set_pos_right.clicked.connect(self.button_set_pos_right_clicked)
    # layout_connection.addWidget(self.button_set_pos_right)
    self.button_disconnect = QPushButton("Disconnect")
    self.button_disconnect.clicked.connect(self.button_disconnect_clicked)
    layout_connection.addWidget(self.button_disconnect)
    layout_connection.setContentsMargins(3,10,3,3)
    gb_connection = QGroupBox("Connection")
    gb_connection.setLayout(layout_connection)
    gb_connection.setFixedWidth(LWIDTH)
    gb_connection.setContentsMargins(0,0,0,0)
    #Initialize
    # self.button_get_pos.setEnabled(False)
    # self.button_set_pos_left.setEnabled(False)
    # self.button_set_pos_right.setEnabled(False)
    return gb_connection