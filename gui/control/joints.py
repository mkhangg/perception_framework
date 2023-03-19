import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtGui
from PyQt5.QtWidgets import QSizePolicy
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
    QGridLayout
)
from matplotlib.widgets import Slider

ljoint_names = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
rjoint_names = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"]
joint_limit = [[-141, 70], [-123, 60], [-173, 173], [-3, 150], [-175, 175], [-90, 120], [-175, 175]]

ljoint_sliders = []
ljoint_labels = []
rjoint_sliders = []
rjoint_labels = []

from control.const import *

def init_joint(self):
    gl = QGridLayout()
    for i in range(7):
        layout = QHBoxLayout()
        label_name = QLabel(ljoint_names[i])
        slider = QSlider(Qt.Horizontal, self)
        #slider.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        #slider.setMaximumWidth(60)
        slider.setRange(joint_limit[i][0], joint_limit[i][1])
        ljoint_sliders.append(slider)
        button_nag = QPushButton("-")
        button_nag.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        button_nag.setMaximumHeight(40)
        button_nag.setMaximumWidth(40)
        button_nag.setAccessibleName(ljoint_names[i]+"-")
        button_nag.clicked.connect(self.button_joint_clicked)
        button_pos = QPushButton("+")
        button_pos.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        button_pos.setMaximumHeight(40)
        button_pos.setMaximumWidth(40)
        button_pos.setAccessibleName(ljoint_names[i]+"+")
        button_pos.clicked.connect(self.button_joint_clicked)
        label_value = QLabel("0")
        ljoint_labels.append(label_value)
        layout.addWidget(label_name)
        layout.addWidget(button_nag)
        layout.addWidget(slider)
        layout.addWidget(label_value)
        layout.addWidget(button_pos)
        gl.addLayout(layout, i, 1)

    for i in range(7):
        layout = QHBoxLayout()
        label_name = QLabel(rjoint_names[i])
        slider = QSlider(Qt.Horizontal, self)
        #slider.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        #slider.setMaximumWidth(60)
        slider.setRange(joint_limit[i][0], joint_limit[i][1])
        rjoint_sliders.append(slider)
        button_nag = QPushButton("-")
        button_nag.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        button_nag.setMaximumHeight(40)
        button_nag.setMaximumWidth(40)
        button_nag.setAccessibleName(rjoint_names[i]+"-")
        button_nag.clicked.connect(self.button_joint_clicked)
        button_pos = QPushButton("+")
        button_pos.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        button_pos.setMaximumHeight(40)
        button_pos.setMaximumWidth(40)
        button_pos.setAccessibleName(rjoint_names[i]+"+")
        button_pos.clicked.connect(self.button_joint_clicked)
        label_value = QLabel("0")
        rjoint_labels.append(label_value)
        layout.addWidget(label_name)
        layout.addWidget(button_nag)
        layout.addWidget(slider)
        layout.addWidget(label_value)
        layout.addWidget(button_pos)
        gl.addLayout(layout, i, 2)
  
    gb =  QGroupBox("Joint Jogging")
    gb.setLayout(gl)
    gb.setFixedWidth(LWIDTH)
    return gb
