from openvino.runtime import Core
import cv2
import numpy as np
import random
import time
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.runtime import Layout, AsyncInferQueue, PartialShape
import torch
from vision.utils import box_utils
from vision.ssd.data_preprocessing import PredictionTransform
import scipy.stats as st

class SSD_OPENVINO(object):
    def __init__(self, model_path, device, pre_api, batchsize, nireq):
        # set the hyperparameters
        self.classes = ['BG', 'cone', 'cube', 'sphere']
        self.batchsize = batchsize
        self.img_size = (300, 300) 
        self.conf_thres = 0.25
        self.iou_threshold =  0.5
        self.sigma = 0.5
        self.candidate_size = 200
        self.model_path = model_path

        self.class_num = len(self.classes)
        self.colors = [(255, 50, 0), (0, 50, 255), (0, 100, 50)]
        self.old_time = time.time()
        self.new_time = time.time()

        self.t_infer1 = time.time()
        self.t_infer2 = time.time()
        self.fps = []

        self.nms_method = 'soft'
        ie = Core()
        self.model = ie.read_model(model_path)
        print('model_path = ', model_path)
        self.input_layer = self.model.input(0)
        new_shape = PartialShape([self.batchsize, 3, self.img_size[0], self.img_size[1]])
        #print('input new_shape = ', new_shape.shape)
        self.model.reshape({self.input_layer.any_name: new_shape})
        self.pre_api = pre_api
        #self.pre_api = False
        if self.pre_api:            
            # Preprocessing API
            ppp = PrePostProcessor(self.model)
            # Declare section of desired application's input format
            ppp.input().tensor() \
                .set_layout(Layout("NHWC")) \
                .set_color_format(ColorFormat.BGR)
            # Here, it is assumed that the model has "NCHW" layout for input.
            ppp.input().model().set_layout(Layout("NCHW"))
            # Convert current color format (BGR) to RGB
            ppp.input().preprocess() \
                .convert_color(ColorFormat.RGB) \
                .scale([255.0, 255.0, 255.0])
            self.model = ppp.build()
            print(f'Dump preprocessor: {ppp}')

        self.device = device
        self.compiled_model = ie.compile_model(model=self.model, device_name=device)
        self.infer_queue = AsyncInferQueue(self.compiled_model, nireq)    

    def draw(self, img_tuple, boxinfo):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def postprocess(self, infer_request, info):
        self.t_infer2 = time.time()
        # print(f'FPS = %2.2f' % (1.0/(self.t_infer2 - self.t_infer1)))
        self.t_infer1 = self.t_infer2

        src_img_list, src_size = info        
        top_k = 10            
        color_frame = src_img_list[0][0]        
        depth_frame = src_img_list[0][1]
        height, width,_ = color_frame.shape
        #print('(height, width = ({height}, {width})')
        for batch_id in range(self.batchsize):            
            #print('postprocess = ', type(infer_request.get_output_tensor), " boxes = ", infer_request.get_output_tensor(0).shape, "scores = ", infer_request.get_output_tensor(1).shape)
            scores = infer_request.get_output_tensor(0).data[batch_id]
            boxes = infer_request.get_output_tensor(1).data[batch_id]            
            #print("boxes = ", type(boxes), ", shape = ", boxes.shape)
            #print("scores = ", type(scores), ", shape = ", scores.shape)
            #'''
            picked_box_probs = []
            picked_labels = []
            for class_index in range(1, scores.shape[1]):
                probs = scores[:, class_index]
                # print('probs 1 = ', probs.shape, '->', probs[0:5])
                mask = probs > self.conf_thres
                probs = probs[mask]
                # print('probs 2 = ', probs.shape, '->', probs[0:5])
                if probs.shape[0] == 0:
                    continue
                #print('boxes {class_index} = ', boxes.shape)
                subset_boxes = boxes[mask, :]
                #print('subset_boxes {class_index} = ', subset_boxes.shape)
                temp = probs.reshape(-1, 1)
                #print("temp = ", type(temp))
                box_probs = torch.cat((torch.tensor(subset_boxes), torch.tensor(temp)), 1)
                box_probs = box_utils.nms(box_probs, self.nms_method,
                                        score_threshold=self.conf_thres,
                                        iou_threshold=self.iou_threshold,
                                        sigma=self.sigma,
                                        top_k=top_k,
                                        candidate_size=self.candidate_size)
                #print('box_probs {class_index} = ', box_probs.shape)
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.size(0))
            #end for
            if not picked_box_probs:
                boxes, labels, probs = torch.tensor([]), torch.tensor([]), torch.tensor([])
            else:
                picked_box_probs = torch.cat(picked_box_probs)
                picked_box_probs[:, 0] *= width
                picked_box_probs[:, 1] *= height
                picked_box_probs[:, 2] *= width
                picked_box_probs[:, 3] *= height
                boxes, labels, probs = picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]                                                   

            self.new_time = time.time()
            fps = 1.0/(self.new_time - self.old_time)
            self.fps.append(fps)
            print(f'fps = {fps}')
            cv2.putText(color_frame, f'FPS = %2.2f' % (fps), (20, 50), 
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, 
                        color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            color = np.random.uniform(0, 255, size = (10, 3))
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                label = f"{self.classes[labels[i]]}: {probs[i]:.2f}"

                i_color = int(labels[i]) - 1
                box = [round(b.item()) for b in box]
                #print(box)
                cv2.rectangle(color_frame, (box[0], box[1]), (box[2], box[3]), self.colors[i_color], 2)

                cv2.putText(color_frame, label,
                            (box[0] - 10, box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            self.colors[i_color%10],
                            2)  # line type
                
            self.old_time = self.new_time
            cv2.imshow("Camera", color_frame)
            cv2.waitKey(1)

            #print('self.fps = ', len(self.fps))
            if len(self.fps) >= 301:
                data = self.fps[1:]
                m = np.mean(data)
                print('Mean FPS = ', round(m, 2))
                interval = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
                print('interval = ', interval)
                file = open('test/fps.txt', 'a')
                tokens = self.model_path.split('/')
                file.write(f'{tokens[-1]}, {self.device}, {round(m, 2)}, {round(interval[0], 2)}, {round(interval[1], 2)}, {round(interval[1] - m, 2)}\n')
                file.close()
                exit(0)
            #'''

    
    def infer_cam(self):
        # Set callback function for postprocess        
        self.infer_queue.set_callback(self.postprocess)
        
        import pyrealsense2 as rs        
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # Configure streaming formats for RGB and Depth images then start streaming
        fps = 30
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        #config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        self.profile = pipeline.start(config)
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        #Skip some first frames
        for _ in range(30):
           frameset = pipeline.wait_for_frames()

        src_img_list = []
        img_list = []    
        while True:            
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            color_frame = np.asanyarray(color_frame.get_data())
            colorizer = rs.colorizer()
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            #Align
            align = rs.align(rs.stream.color)
            frameset = align.process(frameset)
            aligned_depth_frame = frameset.get_depth_frame()
            aligned_depth_frame_view = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())            
                        
            #img = color_frame
            src_size = color_frame.shape[:2]            
            #self.transform = PredictionTransform(self.img_size[0], 0.0, 1.0)
            img = cv2.resize(color_frame, (300, 300), interpolation=cv2.INTER_AREA)
            img = img.astype(dtype=np.float32) #[1, w, h]
            input_image = np.expand_dims(img, 0) #[1, 3, w, h]
            #input_image = self.transform(img)                        
            #print('input_image 1 = ', input_image.shape)  #[3, w, h]
            #input_image = input_image.unsqueeze(0)
            #print('input_image 2 = ', input_image.shape)  #[1, 3, w, h]
            # Batching
            img_list.append(input_image)
            src_img_list.append((color_frame, aligned_depth_frame_view, aligned_depth_frame))            
            if (len(img_list) < self.batchsize):
                continue
            img_batch = np.concatenate(img_list)

            # Do inference
            self.infer_queue.start_async({self.input_layer.any_name: img_batch}, (src_img_list, src_size))                          
            src_img_list = []
            img_list = []                   
            self.infer_queue.wait_all()

    
import argparse
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
devices = ['CPU', 'GPU', 'MYRIAD']

net_id = 1
dev_id = 2

if __name__ == "__main__":
    device = devices[dev_id]
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')    
    args.add_argument('-m', '--model', required=False, type=str, help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-d', '--device', required=False, default=device, type=str,help='Device name.')
    args.add_argument('-p', '--pre_api', required=False, default=True, type=bool, help='Device name.')
    args.add_argument('-bs', '--batchsize', required=False, default=1, type=int, help='Batch size.')
    args.add_argument('-n', '--nireq', required=False, default=1, type=int,help='number of infer request.')
    args = parser.parse_args()
    args.model = f"/home/installer/ssd/models/save/learnlab_models/{model_names[net_id]}.onnx"
    ssd_detector= SSD_OPENVINO(args.model, args.device, args.pre_api, args.batchsize, args.nireq)
    ssd_detector.infer_cam()
