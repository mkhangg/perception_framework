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
    QGroupBox,
    QGridLayout,
    QSizePolicy
)
from control.const import *

def init_nav(self):
    gl_nav_dir = QGridLayout()
    gl_nav_turn = QGridLayout()

    bt_width = 64
    bt_height = 64

    left_button = QPushButton()
    left_button.setStyleSheet("border-image : url(images/left.png);")
    left_button.setMinimumSize(bt_width/2, bt_height/2)
    left_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    left_button.pressed.connect(self.button_base_move_left_pressed)
    left_button.released.connect(self.button_nav_released)

    right_button = QPushButton()
    right_button.setStyleSheet("border-image : url(images/right.png);")
    right_button.setMinimumSize(bt_width/2, bt_height/2)
    right_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    right_button.pressed.connect(self.button_base_move_right_pressed)
    right_button.released.connect(self.button_nav_released)

    top_button = QPushButton()
    top_button.setStyleSheet("border-image : url(images/top.png);")
    top_button.setMinimumSize(bt_width/2, bt_height/2)
    top_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    top_button.pressed.connect(self.button_base_move_forward_pressed)
    top_button.released.connect(self.button_nav_released)
    

    bottom_button = QPushButton()
    bottom_button.setStyleSheet("border-image : url(images/bottom.png);")
    bottom_button.setMinimumSize(bt_width/2, bt_height/2)
    bottom_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    bottom_button.pressed.connect(self.button_base_move_backward_pressed)
    bottom_button.released.connect(self.button_nav_released)


    turn_left_button = QPushButton()
    turn_left_button.setStyleSheet("border-image : url(images/tl.png);")
    turn_left_button.setMinimumSize(bt_width, bt_height)
    turn_left_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    turn_left_button.pressed.connect(self.button_base_turn_left_pressed)
    turn_left_button.released.connect(self.button_nav_released)

    turn_right_button = QPushButton()
    turn_right_button.setStyleSheet("border-image : url(images/tr.png);")
    turn_right_button.setMinimumSize(bt_width, bt_height)
    turn_right_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    turn_right_button.pressed.connect(self.button_base_turn_right_pressed)
    turn_right_button.released.connect(self.button_nav_released)
    

    gl_nav_dir.addWidget(left_button, 1, 0)
    gl_nav_dir.addWidget(right_button, 1, 2)
    gl_nav_dir.addWidget(top_button, 0, 1)
    gl_nav_dir.addWidget(bottom_button, 1, 1)

    gl_nav_turn.addWidget(turn_left_button, 0, 0)
    gl_nav_turn.addWidget(turn_right_button, 0, 1)

    nav_layout = QHBoxLayout()
    nav_layout.addLayout(gl_nav_dir)
    nav_layout.addLayout(gl_nav_turn)

    gb_nav = QGroupBox("Mobility Base Nagivation")
    gb_nav.setLayout(nav_layout)
    gb_nav.setFixedWidth(LWIDTH-150)
    gb_nav.setFixedHeight(100)
    gb_nav.setContentsMargins(0,0,0,0)
    return gb_nav
    