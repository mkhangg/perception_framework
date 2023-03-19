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

def init_coordinate(self):
    label_left_world = QLabel("Left Arm World: ")
    label_right_world = QLabel("Left Arm World: ")
    layout_world_coordinate = QHBoxLayout()
    layout_world_coordinate.addWidget(label_left_world)
    layout_world_coordinate.addWidget(label_right_world)
    gb_world_coordinate = QGroupBox("World Coordinate")
    gb_world_coordinate.setLayout(layout_world_coordinate)
    gb_world_coordinate.setFixedWidth(LWIDTH)
    return gb_world_coordinate