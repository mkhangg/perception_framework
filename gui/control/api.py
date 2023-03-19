import math
import json
import time
from lib.math import rad2deg
from lib.math import deg2rad
from lib.commands import *
from control.joints import  ljoint_names, rjoint_names, ljoint_sliders, rjoint_sliders, ljoint_labels, rjoint_labels
from control.world import lworld_labels, rworld_labels

def send_cmd(sock, cmd):
    if sock != None:
        sock.send(cmd.to_bytes(4, 'little'))
        #data = sock.recv(3).decode()
        time.sleep(0.2)
        #print("ACK = ", data, cmd)

def client_thread_func(sock, window):
    print("client_thread_func")
    while sock != None:
        #LEFT
        send_cmd(sock, GET_LPOS)
        l_bytes = sock.recv(4)
        l = int.from_bytes(l_bytes, "little")
        #print("left l = ", l)
        #time.sleep(0.01)
        joint_data = sock.recv(l).decode("ascii")
        #print(joint_data)
        #time.sleep(0.1)
        joint_dict = json.loads(joint_data)
        #print("======")
        #Joint: LEFT
        #Sliders
        ljoint_sliders[0].setValue(rad2deg(joint_dict['left_s0']))
        ljoint_sliders[1].setValue(rad2deg(joint_dict['left_s1']))
        ljoint_sliders[2].setValue(rad2deg(joint_dict['left_e0']))
        ljoint_sliders[3].setValue(rad2deg(joint_dict['left_e1']))
        ljoint_sliders[4].setValue(rad2deg(joint_dict['left_w0']))
        ljoint_sliders[5].setValue(rad2deg(joint_dict['left_w1']))
        ljoint_sliders[6].setValue(rad2deg(joint_dict['left_w2']))
        #Labels
        ljoint_labels[0].setText(str(rad2deg(joint_dict['left_s0'])))
        ljoint_labels[1].setText(str(rad2deg(joint_dict['left_s1'])))
        ljoint_labels[2].setText(str(rad2deg(joint_dict['left_e0'])))
        ljoint_labels[3].setText(str(rad2deg(joint_dict['left_e1'])))
        ljoint_labels[4].setText(str(rad2deg(joint_dict['left_w0'])))
        ljoint_labels[5].setText(str(rad2deg(joint_dict['left_w1'])))
        ljoint_labels[6].setText(str(rad2deg(joint_dict['left_w2'])))

        #RIGHT
        send_cmd(sock, GET_RPOS)
        l_bytes = sock.recv(4)
        l = int.from_bytes(l_bytes, "little")
        #print("right l = ", l)
        #time.sleep(0.01)
        joint_data = sock.recv(l).decode("ascii")
        #print(joint_data)
        #time.sleep(0.1)
        joint_dict = json.loads(joint_data)
        #Sliders
        rjoint_sliders[0].setValue(rad2deg(joint_dict['right_s0']))
        rjoint_sliders[1].setValue(rad2deg(joint_dict['right_s1']))
        rjoint_sliders[2].setValue(rad2deg(joint_dict['right_e0']))
        rjoint_sliders[3].setValue(rad2deg(joint_dict['right_e1']))
        rjoint_sliders[4].setValue(rad2deg(joint_dict['right_w0']))
        rjoint_sliders[5].setValue(rad2deg(joint_dict['right_w1']))
        rjoint_sliders[6].setValue(rad2deg(joint_dict['right_w2']))
        #Labels
        rjoint_labels[0].setText(str(rad2deg(joint_dict['right_s0'])))
        rjoint_labels[1].setText(str(rad2deg(joint_dict['right_s1'])))
        rjoint_labels[2].setText(str(rad2deg(joint_dict['right_e0'])))
        rjoint_labels[3].setText(str(rad2deg(joint_dict['right_e1'])))
        rjoint_labels[4].setText(str(rad2deg(joint_dict['right_w0'])))
        rjoint_labels[5].setText(str(rad2deg(joint_dict['right_w1'])))
        rjoint_labels[6].setText(str(rad2deg(joint_dict['right_w2'])))

        #time.sleep(0.2)
        #WORLD Positions
        send_cmd(sock, GET_WLPOS)
        l_bytes = sock.recv(4)
        l = int.from_bytes(l_bytes, "little")
        lworld_data = sock.recv(l).decode("ascii")
        #print('GET_WLPOS: ', lworld_data)
        #world_labels[0].setText(lworld_data)
        lworld_data = json.loads(lworld_data)
        ltext = f'LEFT: P(%2.2f, %2.2f, %2.2f), Q(%2.2f, %2.2f, %2.2f, %2.2f)' % (lworld_data[0], lworld_data[1], lworld_data[2], lworld_data[3], lworld_data[4], lworld_data[5], lworld_data[6])
        lworld_labels[0].setText(ltext)
        ltext_rad_deg = f'  Rad(%2.2f, %2.2f, %2.2f), Deg(%2.2f, %2.2f, %2.2f)' % (lworld_data[7], lworld_data[8], lworld_data[9], rad2deg(lworld_data[7]), rad2deg(lworld_data[8]), rad2deg(lworld_data[9]))
        lworld_labels[1].setText(ltext_rad_deg)
        

        send_cmd(sock, GET_WRPOS)
        l_bytes = sock.recv(4)
        l = int.from_bytes(l_bytes, "little")
        rworld_data = sock.recv(l).decode("ascii")
        #print('GET_WRPOS: ', rworld_data)
        #world_labels[1].setText(rworld_data)
        rworld_data = json.loads(rworld_data)
        rtext = f'RIGHT: P(%2.2f, %2.2f, %2.2f), Q(%2.2f, %2.2f, %2.2f, %2.2f)' %(rworld_data[0], rworld_data[1], rworld_data[2], rworld_data[3], rworld_data[4], rworld_data[5], rworld_data[6])
        rworld_labels[0].setText(rtext)
        rtext_rad_deg = f'  Rad(%2.2f, %2.2f, %2.2f), Deg(%2.2f, %2.2f, %2.2f)' % (rworld_data[7], rworld_data[8], rworld_data[9], rad2deg(rworld_data[7]), rad2deg(rworld_data[8]), rad2deg(rworld_data[9]))
        rworld_labels[1].setText(rtext_rad_deg)



def get_pos(sock, window):
    sock.send("GetPosL".encode("ascii"))
    l_bytes = sock.recv(4)
    l = int.from_bytes(l_bytes, "little")
    #print("left l = ", l)
    left_joint_data = sock.recv(l).decode("ascii")
    #print(left_joint_data)
    left_joint_dict = json.loads(left_joint_data)
    #print("======")
    #Joint: LEFT
    window.slider_left_s0.setValue(rad2deg(left_joint_dict['left_s0']))
    window.slider_left_s1.setValue(rad2deg(left_joint_dict['left_s1']))
    window.slider_left_e0.setValue(rad2deg(left_joint_dict['left_e0']))
    window.slider_left_e1.setValue(rad2deg(left_joint_dict['left_e1']))
    window.slider_left_w0.setValue(rad2deg(left_joint_dict['left_w0']))
    window.slider_left_w1.setValue(rad2deg(left_joint_dict['left_w1']))
    window.slider_left_w2.setValue(rad2deg(left_joint_dict['left_w2']))

    #Joint: RIGHT
    sock.send("GetPosR".encode("ascii"))
    l_bytes = sock.recv(4)
    l = int.from_bytes(l_bytes, "little")
    #print("right l = ", l)
    right_joint_data = sock.recv(l).decode("ascii")
    #print(right_joint_data)
    right_joint_dict = json.loads(right_joint_data)
    window.slider_right_s0.setValue(rad2deg(right_joint_dict['right_s0']))
    window.slider_right_s1.setValue(rad2deg(right_joint_dict['right_s1']))
    window.slider_right_e0.setValue(rad2deg(right_joint_dict['right_e0']))
    window.slider_right_e1.setValue(rad2deg(right_joint_dict['right_e1']))
    window.slider_right_w0.setValue(rad2deg(right_joint_dict['right_w0']))
    window.slider_right_w1.setValue(rad2deg(right_joint_dict['right_w1']))
    window.slider_right_w2.setValue(rad2deg(right_joint_dict['right_w2']))


def set_pos(sock, window, left_arm=True):
    if left_arm == True:
        print("button_set_pos_left_clicked")
        left_joint_dict = {}
        left_joint_dict['left_s0'] = deg2rad(window.slider_left_s0.value())
        left_joint_dict['left_s1'] = deg2rad(window.slider_left_s1.value())
        left_joint_dict['left_e0'] = deg2rad(window.slider_left_e0.value())
        left_joint_dict['left_e1'] = deg2rad(window.slider_left_e1.value())
        left_joint_dict['left_w0'] = deg2rad(window.slider_left_w0.value())
        left_joint_dict['left_w1'] = deg2rad(window.slider_left_w1.value())
        left_joint_dict['left_w2'] = deg2rad(window.slider_left_w2.value())
        #print("left_joint_dict = ", left_joint_dict)
        sock.send("SetPosL".encode("ascii"))
        left_joint_json = json.dumps(left_joint_dict).encode("ascii")
        print("left_joint_json = ", left_joint_json)
        l = int(len(left_joint_json))
        print(" >>> left l = ", l)
        sock.send( l.to_bytes(4, 'little'))
        sock.send(left_joint_json)
        ack = sock.recv(3).decode("ascii")
        print("Set left joint = ", ack)
    else:
        right_joint_dict = {}
        right_joint_dict['right_s0'] = deg2rad(window.slider_right_s0.value())
        right_joint_dict['right_s1'] = deg2rad(window.slider_right_s1.value())
        right_joint_dict['right_e0'] = deg2rad(window.slider_right_e0.value())
        right_joint_dict['right_e1'] = deg2rad(window.slider_right_e1.value())
        right_joint_dict['right_w0'] = deg2rad(window.slider_right_w0.value())
        right_joint_dict['right_w1'] = deg2rad(window.slider_right_w1.value())
        right_joint_dict['right_w2'] = deg2rad(window.slider_right_w2.value())
        #print(right_joint_dict)
        sock.send("SetPosR".encode("ascii"))
        right_joint_json = json.dumps(right_joint_dict).encode("ascii")
        print("right_joint_json = ", right_joint_json)
        l = int(len(right_joint_json))
        print(" >>> right l = ", l)
        sock.send( l.to_bytes(4, 'little'))
        sock.send(right_joint_json)
        ack = sock.recv(3).decode("ascii")
        print("Set right joint = ", ack)
