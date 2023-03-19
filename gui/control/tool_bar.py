from PyQt5.QtWidgets import *
import sys

def init_tool_bar(self):    
    # Create pyqt toolbar
    toolBar = QToolBar()
    layout = QGridLayout()
    layout.addWidget(toolBar)

    # Add buttons to toolbar
    toolButton = QToolButton()
    toolButton.setText("File")
    toolButton.setCheckable(True)
    toolButton.setAutoExclusive(True)
    toolBar.addWidget(toolButton)
    toolButton = QToolButton()
    toolButton.setText("Help")
    toolButton.setCheckable(True)
    toolButton.setAutoExclusive(True)
    toolBar.addWidget(toolButton)
    return toolBar