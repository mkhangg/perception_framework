import sys
sys

from PyQt5 import QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import socket
import threading
from control.connection import *
from control.coordinate import *
from control.robot_control import *
from control.hands_control import *
from control.world import *
from control.joints import *
from control.nav import *
from control.api import *
from control.tool_bar import *
from control.menu_bar import *
from lib.commands import *
from control.program import *


client_running = False
port = 3000

class MainWindow(QMainWindow):
    def __init__(self,parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('Rosie Controller')
        self.setWindowIcon(QtGui.QIcon('images/baxter.png'))
        #self.setGeometry(0, 0, 500, 450)
        init_menu_bar(self)
        # self.addToolBar(init_tool_bar(self))

        self.sock = None
        grid = QGridLayout()
        grid.addWidget(init_connection(self), 1, 1)
        grid.addWidget(init_robot_control(self), 2, 1)
        grid.addWidget(init_hands_control(self), 3, 1)
        # grid.addWidget(init_coordinate(self), 4, 1)
        grid.addWidget(init_world(self), 4, 1)
        grid.addWidget(init_joint(self), 5, 1)
        

        grid.addWidget(init_programs(self), 1, 2, 6, 1)
        #grid.addWidget(init_nav(self), 6, 2)
        
        gb = QGroupBox()
        gb.setLayout(grid)
        self.setCentralWidget(gb)
        
    def button_connect_clicked(self):
        global client_running
        ip_address = self.text_ip.text()
        print("button_connect_clicked = ", ip_address)
        self.sock = socket.socket()
        self.sock.setblocking(1)	
        try:
            self.sock.connect((ip_address, port))  # connect to the server
            print("Connected!")
            #get_pos(self.sock, self)
            self.button_connect.setEnabled(False)
            # self.button_get_pos.setEnabled(True)
            # self.button_set_pos_left.setEnabled(True)
            # self.button_set_pos_right.setEnabled(True)
            

            client_running = True
            client_thread = threading.Thread(
                target=client_thread_func,
                args=(self.sock, self)
            )
            client_thread.start()
        except:
            self.sock = None
            print("Connect to server error!")

    def button_joint_clicked(self):
        button = self.sender()
        name = button.accessibleName()
        #Decode which button from name 
        #Then convert it into the final command code
        cmd = 0
        prefix = name[0:4]
        len_name = len(name)
        join_name = name[0:len_name-1]
        sign = name[len_name-1]
        sign_num = 0 if sign == "-" else 1
        if prefix == "left":
            joint_index = ljoint_names.index(join_name)
            cmd = LEFT_S0_NAG + joint_index*2 + sign_num
            print(f"%s $ %s $ %s: (%d,  %d) > cmd = %d" % ( name, join_name, sign, joint_index, sign_num, cmd))
        else:
            joint_index = rjoint_names.index(join_name)
            cmd = RIGHT_S0_NAG + joint_index*2 + sign_num
            print(f"%s $ %s $ %s: (%d,  %d) > cmd = %d" % ( name, join_name, sign, joint_index, sign_num, cmd))
       
        #Send command
        
        send_cmd(self.sock, cmd)
    
    def button_get_pos_clicked(self):
        get_pos(self.sock, self)

    def button_set_pos_left_clicked(self):
        set_pos(self.sock, self, left_arm=True)


    def button_set_pos_right_clicked(self):
        set_pos(self.sock, self, left_arm=True)

    def button_disconnect_clicked(self):
        print("button_disconnect_clicked")
        if self.sock != None:
            self.sock.close()
        self.button_connect.setEnabled(True)
        # self.button_get_pos.setEnabled(False)
        # self.button_set_pos_left.setEnabled(False)

    #Robot Actions
    def button_enable_robot_clicked(self):
        send_cmd(self.sock, ENBABLE_ROBOT)

    def button_disable_robot_clicked(self):
        send_cmd(self.sock, DISABLE_ROBOT)
    
    def button_tuck_robot_clicked(self):
        print("tuck:")
        send_cmd(self.sock, TUCK_ROBOT)

    def button_untuck_robot_clicked(self):
        print("untuck:")
        send_cmd(self.sock, UNTUCK_ROBOT)

    #Hands
    def button_calibrate_hands_clicked(self):
        send_cmd(self.sock, CALIBRATE_HANDS)

    def button_zero_hands_clicked(self):
        send_cmd(self.sock, ZERO_HANDS)

    def button_open_left_hand_clicked(self):
        send_cmd(self.sock, OPEN_LEFT_HAND)

    def button_close_left_hand_clicked(self):
        send_cmd(self.sock, CLOSE_LEFT_HAND)

    def button_open_right_hand_clicked(self):
        send_cmd(self.sock, OPEN_RIGHT_HAND)

    def button_close_right_hand_clicked(self):
        send_cmd(self.sock, CLOSE_RIGHT_HAND)

    #Base commands
    def button_base_move_left_pressed(self):
        
        send_cmd(self.sock, BASE_MOVE_LEFT)

    def button_base_move_right_pressed(self):
        
        send_cmd(self.sock, BASE_MOVE_RIGHT)

    def button_base_move_forward_pressed(self):
        
        send_cmd(self.sock, BASE_MOVE_FORWARD)

    def button_base_move_backward_pressed(self):
        
        send_cmd(self.sock, BASE_MOVE_BACKWARD)
    
    def button_base_turn_left_pressed(self):
        
        send_cmd(self.sock, BASE_TURN_LEFT)

    def button_base_turn_right_pressed(self):
        
        send_cmd(self.sock, BASE_TURN_RIGHT)

    def button_nav_released(self):
        
        print("button_nav_released")
        send_cmd(self.sock, BASE_STOP)
    
    def resizeEvent(self, event):
        #print("Window has been resized = ", event.size())
        pass
        
    #Menubar
    def openCall(self):
        print('Open')

    def newCall(self):
        print('New')

    def exitCall(self):
        print('Exit app')

    def clickMethod(self):
        print('PyQt')

    #Program
    def program_clicked(self, qmodelindex):
        item = self.pitems.currentItem()
        print(item.text())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())