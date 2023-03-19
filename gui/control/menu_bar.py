import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction
from PyQt5.QtCore import QSize    
from PyQt5.QtGui import QIcon

def init_menu_bar(self):
    # Add button widget
    # pybutton = QPushButton('Pyqt', self)
    # pybutton.clicked.connect(self.clickMethod)
    # pybutton.resize(100,32)
    # pybutton.move(130, 30)        
    # pybutton.setToolTip('This is a tooltip message.')  

    # Create new action
    newAction = QAction(QIcon('new.png'), '&New', self)        
    newAction.setShortcut('Ctrl+N')
    newAction.setStatusTip('New document')
    newAction.triggered.connect(self.newCall)

    # Create new action
    openAction = QAction(QIcon('open.png'), '&Open', self)        
    openAction.setShortcut('Ctrl+O')
    openAction.setStatusTip('Open document')
    openAction.triggered.connect(self.openCall)

    # Create exit action
    exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
    exitAction.setShortcut('Ctrl+Q')
    exitAction.setStatusTip('Exit application')
    exitAction.triggered.connect(self.exitCall)

    # Create menu bar and add action
    menuBar = self.menuBar()
    fileMenu = menuBar.addMenu('&File')
    fileMenu.addAction(newAction)
    fileMenu.addAction(openAction)
    fileMenu.addAction(exitAction)

    fileMenu = menuBar.addMenu('&Program List')
    pMenu = menuBar.addMenu('&View')
    helpMenu = menuBar.addMenu('&Help')