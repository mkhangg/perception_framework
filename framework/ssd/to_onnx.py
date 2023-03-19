import os
from click import UsageError
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
import torch
import onnx


model_names = [
    'mb1_ssd_freeze',       #0
    'mb1_ssd_retrain', #1
    'mb1_ssd_scratch', #2
    'mb2_ssd_lite_freeze', #3
    'mb2_ssd_lite_retrain', #4
    'mb2_ssd_lite_scratch', #5
    'sq_ssd_lite_freeze', #6
    'sq_ssd_lite_retrain', #7
    'sq_ssd_lite_scratch', #8
    'vgg16_ssd_freeze', #9
    'vgg16_ssd_retrain', #10
    'vgg16_ssd_scratch', #11
]

net_id = 11

if __name__ == "__main__":	
	model_path = f'models/save/learnlab_models/{model_names[net_id]}.pth'
	timer = Timer()
	label_path = "models/voc-model-labels.txt"
	class_names = [name.strip() for name in open(label_path).readlines()]	

	tokens = model_names[net_id].split('_')
	tokens_len = len(tokens)
	net_type = ''
	for i in range(tokens_len - 1):
		net_type = net_type + tokens[i] + '_'
	net_type = net_type[0:-1]
	print(net_type)
	#exit()
	if net_type == 'vgg16_ssd':
		net = create_vgg_ssd(len(class_names), is_test=True)
	elif net_type == 'mb1_ssd':
		net = create_mobilenetv1_ssd(len(class_names), is_test=True)
	elif net_type == 'mb1_ssd_lite':
		net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'mb2_ssd_lite':
		net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'sq_ssd_lite':
		net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
	else:
		print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
		sys.exit(1)

	print(f'Load model: {model_path}')	
	net.load(model_path)
	net.eval()
	print("Loading Trained Model is Done!")	
	input_size = 300
	dummy_input = torch.randn(1,  3, input_size,  input_size)  
	out_path = model_path[0:-4] + '.onnx'	
	torch.onnx.export(
		net,
		dummy_input,
		out_path,
		verbose=False,
		opset_version=12)		
	print(f'Model has been converted to ONNX at : {out_path}') 