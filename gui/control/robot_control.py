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
def init_robot_control(self):
    self.button_enable_robot = QPushButton("Enable Robot")
    self.button_enable_robot.clicked.connect(self.button_enable_robot_clicked)
    self.button_disable_robot = QPushButton("Disable Robot")
    self.button_disable_robot.clicked.connect(self.button_disable_robot_clicked)
    self.button_tuck_robot = QPushButton("Tuck/Fold Robot")
    self.button_tuck_robot.clicked.connect(self.button_tuck_robot_clicked)
    self.button_untuck_robot = QPushButton("Untuck/Unfold Robot")
    self.button_untuck_robot.clicked.connect(self.button_untuck_robot_clicked)
    layout_robot_control = QHBoxLayout()
    layout_robot_control.addWidget(self.button_enable_robot)
    layout_robot_control.addWidget(self.button_disable_robot)
    layout_robot_control.addWidget(self.button_tuck_robot)
    layout_robot_control.addWidget(self.button_untuck_robot)
    layout_robot_control.setContentsMargins(3,12,3,3)
    gb_robot_control = QGroupBox("Robot Controls")
    gb_robot_control.setLayout(layout_robot_control)
    gb_robot_control.setFixedWidth(LWIDTH)
    gb_robot_control.setContentsMargins(0,0,0,0)
    return gb_robot_control