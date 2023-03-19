from PyQt5.QtWidgets import *
import control.syntax  as syntax
from PyQt5 import QtWidgets
from control.const import *
from control.nav import init_nav

def init_programs(self):
    layout = QGridLayout()
    self.pitems = QListWidget()
    self.pitems.insertItem(0, "init_robot.py")
    self.pitems.insertItem(1, "demo_motion.py")
    self.pitems.insertItem(2, "demo_tracking.py")
    self.pitems.insertItem(3, "demo_building_map.py")
    self.pitems.insertItem(4, "demo_navigation.py")
    self.pitems.clicked.connect(self.program_clicked)
    self.pitems.setFixedWidth(190)

    self.editor = QtWidgets.QPlainTextEdit()
    self.editor.setStyleSheet("""QPlainTextEdit{
        font-family:'Consolas'; 
        color: black; 
        background-color: white;}""")
    highlight = syntax.PythonHighlighter(self.editor.document())
    #editor.show()

    # Load syntax.py into the editor for demo purposes
    infile = open('p.py', 'r')
    self.editor.setPlainText(infile.read())
    #layout.addWidget(self.pitems, 1, 1)
    layout.addWidget(self.editor, 0, 0)
    layout.addWidget(init_nav(self))

    gb= QGroupBox("Program: demo.py")
    gb.setLayout(layout)
    return gb