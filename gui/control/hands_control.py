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

def init_hands_control(self):
    self.button_clibrate_hands = QPushButton("Calib Hands")
    self.button_clibrate_hands.clicked.connect(self.button_calibrate_hands_clicked)
    self.button_zero_hands = QPushButton("Zero Hands")
    self.button_zero_hands.clicked.connect(self.button_zero_hands_clicked)
    self.button_open_left_hand = QPushButton("Open LHand")
    self.button_open_left_hand.clicked.connect(self.button_open_left_hand_clicked)
    self.button_close_left_hand = QPushButton("Close LHand")
    self.button_close_left_hand.clicked.connect(self.button_close_left_hand_clicked)
    self.button_open_right_hand = QPushButton("Open RHand")
    self.button_open_right_hand.clicked.connect(self.button_open_right_hand_clicked)
    self.button_close_right_hand = QPushButton("Close RHand")
    self.button_close_right_hand.clicked.connect(self.button_close_right_hand_clicked)
    gb_hands_control = QGroupBox("Hand Controls")  #layout_hands_control
    layout_hands_control = QHBoxLayout()
    layout_hands_control.addWidget(self.button_clibrate_hands)
    layout_hands_control.addWidget(self.button_zero_hands)
    layout_hands_control.addWidget(self.button_open_left_hand)
    layout_hands_control.addWidget(self.button_close_left_hand)
    layout_hands_control.addWidget(self.button_open_right_hand)
    layout_hands_control.addWidget(self.button_close_right_hand)
    layout_hands_control.setContentsMargins(3,12,3,3)
    gb_hands_control.setLayout(layout_hands_control)
    gb_hands_control.setFixedWidth(LWIDTH)
    gb_hands_control.setContentsMargins(0,0,0,0)
    return gb_hands_control