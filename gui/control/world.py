from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QGroupBox,
)

from control.const import *
lworld_labels = []
rworld_labels = []

def init_world(self):
    lworld = QLabel("Left Hand World Postion")
    lworld_rad_deg = QLabel("Rad-deg")
    lworld_labels.append(lworld)
    lworld_labels.append(lworld_rad_deg)
    
    rworld = QLabel("Right Hand World Postion")
    rworld_rad_deg = QLabel("Rad-deg")
    rworld_labels.append(rworld)
    rworld_labels.append(rworld_rad_deg)

    layout_r1 = QVBoxLayout()
    layout_r1.addWidget(lworld)
    layout_r1.addWidget(lworld_rad_deg)
    layout_r2 = QVBoxLayout()
    layout_r2.addWidget(rworld)
    layout_r2.addWidget(rworld_rad_deg)

    layout = QHBoxLayout()
    layout.addLayout(layout_r1)
    layout.addLayout(layout_r2)
    layout.setContentsMargins(3, 12, 3, 3)
    gb =  QGroupBox("World Position")
    gb.setLayout(layout)
    gb.setFixedWidth(LWIDTH)
    gb.setContentsMargins(0, 0, 0, 0)
    return gb