from openvino.runtime import Core
import cv2
import numpy as np
import random
import time
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.runtime import Layout, AsyncInferQueue, PartialShape

import scipy.stats as st 
from cv_bridge import CvBridge
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

br = CvBridge()
pub_baxter = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=1)
rospy.init_node('camera_detector', anonymous=True)

# h = imgsz[0]
# w = imgsz[1]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('videos/out.avi', fourcc, 22, (1024, 281))

import signal
import sys
import tf
import threading
import time

trans = [0, 0, 0]
def get_trans():
    print('get_trans thread!')
    global trans
    while True:
        listener = tf.TransformListener()
        try:
            #base_link
            (trans,rot) = listener.lookupTransform('/base_link_d435', '/right_hand', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #print('Exception!!!')
            pass
            #trans = [0, 0, 0] 
            #rot = [0, 0, 0, 0]
        print('Thread trans= ', trans)
        time.sleep(1)

trans_thread = threading.Thread(
                target=get_trans,
                args=()
            )
#trans_thread.start()

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    out.release()  
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

class YOLOV7_OPENVINO(object):
    def __init__(self, model_path, device, pre_api, batchsize, nireq, im_size):
        # set the hyperparameters
        self.classes = ['cone', 'cube', 'sphere']
        self.batchsize = batchsize
        self.img_size = (im_size, im_size) 
        print('self.img_size = ', self.img_size)
        self.conf_thres = 0.8
        self.iou_thres = 0.85
        self.class_num = len(self.classes)
        self.colors = [(255, 50, 0), (0, 50, 255), (0, 100, 50)]
        self.stride = [8, 16, 32]
        self.anchor_list = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
        self.anchor = np.array(self.anchor_list).astype(float).reshape(3, -1, 2)
        area = self.img_size[0] * self.img_size[1]
        self.size = [int(area / self.stride[0] ** 2), int(area / self.stride[1] ** 2), int(area / self.stride[2] ** 2)]
        self.feature = [[int(j / self.stride[i]) for j in self.img_size] for i in range(3)]
        self.fps = []

        ie = Core()
        self.model = ie.read_model(model_path)
        self.input_layer = self.model.input(0)
        new_shape = PartialShape([self.batchsize, 3, self.img_size[0], self.img_size[1]])
        self.model.reshape({self.input_layer.any_name: new_shape})
        self.pre_api = pre_api
        if (self.pre_api == True):
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
        self.old_time = time.time()
        self.new_time = time.time()

        #Calib
        self.count = 0
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        # resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # add border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def nms(self, prediction, conf_thres, iou_thres):
        predictions = np.squeeze(prediction[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > conf_thres]
        obj_conf = obj_conf[obj_conf > conf_thres]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        valid_scores = scores > conf_thres
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.xywh2xyxy(predictions[:, :4])

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)

        return boxes[indices], scores[indices], class_ids[indices]

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, img0_shape, coords, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        # gain  = old / new
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            padding = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]
            padding = ratio_pad[1]
        coords[:, [0, 2]] -= padding[0]  # x padding
        coords[:, [1, 3]] -= padding[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):        
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf1 = max(tl - 1, 1)  # font thicknessimg_view
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf1)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, (255,255,255), thickness=tf1, lineType=cv2.LINE_AA)
    
    def draw(self, img_tuple, boxinfo):
        color_frame = img_tuple[0]        
        depth_frame = img_tuple[1]
        aligned_depth_frame = img_tuple[2]

    
        depth = np.asanyarray(aligned_depth_frame.get_data())
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
       

        self.new_time = time.time()
        fps = 1.0/(self.new_time - self.old_time)
        self.fps.append(fps)
        if len(self.fps) > 301:
            data = self.fps[1:]
            m = np.mean(data)
            print('Mean fps = ', m)
            interval = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
            print('interval = ', interval)
            print(f'{"YOLOv7"}, {self.device}, {round(m, 2)}, {round(interval[0], 2)}, {round(interval[1], 2)}, {round(interval[1] - m, 2)}\n')
            exit()
        cv2.putText(color_frame, f'FPS = %2.2f' % (fps), (20, 50), 
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, 
                        color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        for xyxy, conf, cls in boxinfo:
            self.plot_one_box(xyxy, color_frame, label=f'{self.classes[int(cls)]}:{conf:.2f}%',  color=self.colors[int(cls)], line_thickness=2)            
            x1 = int(xyxy[0].item())
            y1 = int(xyxy[1].item())
            x2 = int(xyxy[2].item())
            y2 = int(xyxy[3].item())
            x_center = int((x1 + x2)/2.0)
            y_center = int((y1 + y2)/2.0)
            box_size = 15
            depth2 = depth[y_center-box_size:y_center+box_size,x_center-box_size:x_center+box_size].astype(float)
            #print("depth_scale = ", depth_scale, ",depth = ", depth.shape, ", depth2 = ", depth2.shape)
            depth2 = depth2 * depth_scale
            dist,_,_,_ = cv2.mean(depth2)            
            self.plot_one_box(xyxy, depth_frame, label=f'{self.classes[int(cls)]}:{dist:.2f}m',  color=self.colors[int(cls)], line_thickness=2)
        self.old_time = self.new_time
        
        # print('trans = ', trans)
        # cv2.putText(color_frame, f'Ground Truth Distance = %2.2f' % (trans[0]), (20, 100), 
                        # fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, 
                        # color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
        img_view = np.hstack((color_frame, depth_frame))
        #ratio = 1024/img_view.shape[0]
        # resized = cv2.resize(img_view, (1024, int(600/ratio)), interpolation = cv2.INTER_AREA)
        # img = np.zeros((600, 1024, 3), dtype = np.uint8) #black image
        # h_offset = int((600 - resized.shape[0])/2)        
        # img[h_offset:h_offset+resized.shape[0], 0:resized.shape[1],  0:3] = resized
        # ros_frame2 = br.cv2_to_imgmsg(img, encoding="bgr8")
        # pub_baxter.publish(ros_frame2)
        cv2.imshow(f'Caliration on device: {self.device}', img_view) 
        cv2.waitKey(1)
        #out.write(resized)


    def postprocess(self, infer_request, info):
        src_img_list, src_size = info
        for batch_id in range(self.batchsize):
            output = []
            # Get the each feature map's output data
            output.append(self.sigmoid(infer_request.get_output_tensor(0).data[batch_id].reshape(-1, self.size[0]*3, 5+self.class_num)))
            output.append(self.sigmoid(infer_request.get_output_tensor(1).data[batch_id].reshape(-1, self.size[1]*3, 5+self.class_num)))
            output.append(self.sigmoid(infer_request.get_output_tensor(2).data[batch_id].reshape(-1, self.size[2]*3, 5+self.class_num)))
            
            # Postprocessing
            grid = []
            for _, f in enumerate(self.feature):
                grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

            result = []
            for i in range(3):
                src = output[i]
                xy = src[..., 0:2] * 2. - 0.5
                wh = (src[..., 2:4] * 2) ** 2
                dst_xy = []
                dst_wh = []
                for j in range(3):
                    dst_xy.append((xy[:, j * self.size[i]:(j + 1) * self.size[i], :] + grid[i]) * self.stride[i])
                    dst_wh.append(wh[:, j * self.size[i]:(j + 1) *self.size[i], :] * self.anchor[i][j])
                src[..., 0:2] = np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1)
                src[..., 2:4] = np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1)
                result.append(src)

            results = np.concatenate(result, 1)
            boxes, scores, class_ids = self.nms(results, self.conf_thres, self.iou_thres)
            img_shape = self.img_size
            self.scale_coords(img_shape, src_size, boxes)

            # Draw the results
            self.draw(src_img_list[batch_id], zip(boxes, scores, class_ids))

    def calib_cam(self, img):
        grid_size = (7, 7)
        cell_size = 25 #mm
        z_w = 500 # mm world
        N_CALIB_LOOP = 20        

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2)
        #print("objp = ", objp.shape)
        # Arrays to store object points and image points from all the images.
        
        #count = 0
        k = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                objp[k, 0] = cell_size*i
                objp[k, 1] = cell_size*j
                objp[k, 2] = 0
                k += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        # If found, add object points, image points (after refining them)
        if ret == True:            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(img, grid_size, corners2, ret)
            #print("corners2 = ", corners2.shape)
            print("count     = ", self.count)
            #print("===========================")
            print("corners2 0 = ", corners2[0])
            print("corners2 1 = ", corners2[1])

            #print(type(corners2[0][0][0]), corners2[0][0][0])
            x0, y0 = corners2[0][0][0], corners2[0][0][1]
            x1, y1 = corners2[1][0][0], corners2[1][0][1]
            print(x0, y0, x1, y1)
            
            if abs(x1 - x0) >= 15 and abs(y1 - y0) <= 3:
            #if abs(x1 - x0) < 5 and abs(y1 - y0) >= 25:
            #if True:
                self.count += 1
                self.objpoints.append(objp)
                self.imgpoints.append(corners2)
                print("objpoints = ", len(self.objpoints))
          
            #cv2.imshow("Calibrating camera...", img)
            #k = cv2.waitKey(1)
        return img, gray
    
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
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        #config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        self.profile = pipeline.start(config)
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        fx = float(intr.fx) # Focal length of x
        fy = float(intr.fy) # Focal length of y
        ppx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
        ppy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
        axs = 0.0 # Axis skew
        mtx = np.array([[fx, axs, ppx],
                        [0.0, fy, ppy],
                        [0.0, 0.0, 1.0]])

        #Skip some first frames
        for _ in range(30):
           frameset = pipeline.wait_for_frames()

        src_img_list = []
        img_list = []    

        SHOW_IMAGE = 0
        GET_INTRINSIC_MATRIX = 1        
        TEST_CHESSBOARD = 2
        GET_EXTRINSIC_MATRIX = 3
        
        step = SHOW_IMAGE
        flag = False
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

            #Calibrate camera                                    
            if step == GET_INTRINSIC_MATRIX:
                if self.count < 3:
                    img, gray = self.calib_cam(color_frame)
                    cv2.imshow("Calibrate 1...", img)
                    cv2.waitKey(1)                    
                else:
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
                    print("ret 1 = ", ret)
                    print(mtx)
                    print(dist)                    
                    #print(rvecs)
                    np.save("cam_matrix/mtx4.npy", mtx)
                    np.save("cam_matrix/dist4.npy", dist)                    

                    print('GET_INTRINSIC_MATRIX: DONE!')
                    self.count = 0
                    self.objpoints = []
                    self.imgpoints = []
                    step = TEST_CHESSBOARD
                    #break
            elif step == TEST_CHESSBOARD:
                #mtx = np.load("cam_matrix/mtx4.npy")
                #dist = np.load("cam_matrix/dist4.npy")

                #mtx = np.array([[614.8515625, 0.0, 322.1319885253906], 
                #                [0.0, 615.16162109375, 243.16213989257812],
                #                [0.0, 0.0, 1.0]])
                dist = np.array([0, 0, 0, 0, 0])
                #Get extrinsic matrix for at chessboard
                if self.count < 3:
                    img, gray = self.calib_cam(color_frame)
                    print("TEST_CHESSBOARD: = %d" %(self.count))
                    cv2.imshow("Calibrate 2...", img)
                    cv2.waitKey(1)                 
                elif self.count < 5:
                    self.count += 1                    
                    if flag == False:
                        print("TEST_CHESSBOARD: GET EXTRINSIC = %d" %(self.count))
                        flag = True
                        z_w = 750 #mm
                        np_obj = np.array(self.objpoints).reshape(-1, 3)
                        np_img = np.array(self.imgpoints).reshape(-1, 2)                                           
                        for i in range(np_obj.shape[0]):
                            np_obj[i][2] = z_w

                        ret, rvec1, tvec1 = cv2.solvePnP(np_obj, np_img, mtx, dist)
                        R_mtx, jac = cv2.Rodrigues(rvec1)
                        print("===============")
                        print("mtx = ", mtx)
                        print("R_mtx = ", R_mtx)
                        inv_cam = np.linalg.inv(mtx)
                        inv_r = np.linalg.inv(R_mtx)
                        tran = tvec1                        
                    else:
                        print("\nTEST_CHESSBOARD: TEST = %d" %(self.count))
                        grid_size = (7, 7)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                        ret, corners = cv2.findChessboardCorners(gray, grid_size, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
                        if ret:
                            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                            x0, y0 = corners2[0][0][0], corners2[0][0][1]
                            x1, y1 = corners2[1][0][0], corners2[1][0][1]
                            #print(x0, y0, x1, y1)
                            # print("count     = ", self.count)
                            # print("corners2 0 = ", corners2[0])
                            # print("corners2 1 = ", corners2[1])
                            
                            x_w, y_w = [], []
                            x_t, y_t = [], []
                            dist = []
                            import math
                            if abs(x1 - x0) >= 15 and abs(y1 - y0) <= 3:
                            #if abs(x1 - x0) < 5 and abs(y1 - y0) >= 25:
                                #print("===")
                                for i in range(7*7):
                                    #k = i*7 + i                
                                    k = i
                                    center = corners2[k,:,:]
                                    #print("center = ", center)
                                    cv2.drawChessboardCorners(color_frame, (1,1), center, ret) #Draw center corner
                                    u, v = int(center[0][0]), int(center[0][1])
                                    #world = calculate_XYZ(u, v, inv_cam, inv_r, tran)                                    
                                    scale = 1100
                                    uv_1 = np.array([[u,v,1]], dtype=np.float32)
                                    uv_1 = uv_1.T
                                    suv_1 = scale*uv_1
                                    xyz_c = inv_cam.dot(suv_1)
                                    xyz_c = xyz_c - tran
                                    XYZ = inv_r.dot(xyz_c)
                                    x_w.append(XYZ[0])
                                    y_w.append(XYZ[1])
                                    x_t.append(np_obj[k][0])
                                    y_t.append(np_obj[k][1])
                                    dist.append(math.dist((XYZ[0], XYZ[1]), (np_obj[k][0], np_obj[k][1])))
                                    print('point_%02d (%d, %d) at World (%2.2f, %2.2f, %2.2f). Ground True = (%03.2f, %03.2f, %03.2f)' %(k, u, v, *XYZ, *np_obj[k]))
                            
                            print("Average Distance between Estimation and Ground True = ", np.mean(dist))
                            import matplotlib.pyplot as plt
                            plt.plot(x_w, y_w, '*')
                            plt.plot(x_t, y_t, 'o')
                            plt.xlabel('X (mm)')
                            plt.ylabel('Y (mm)')
                            plt.legend(['Estimation', 'Ground True'])
                            plt.show()
                            
                else:
                    self.count = 0     
                    self.objpoints = []
                    self.imgpoints = []               
                    break
                cv2.imshow("Calibrate 2...", img)
                cv2.waitKey(1)  

            elif step == GET_EXTRINSIC_MATRIX or step == SHOW_IMAGE: 
                #Detect object and calculate depth
                #cv2.destroyAllWindows()
                img = self.letterbox(color_frame, self.img_size)
                src_size = color_frame.shape[:2]
                img = img.astype(dtype=np.float32)
                # Preprocessing
                #print('img = ', img.shape)
                input_image = np.expand_dims(img, 0) #[1, 3, w, h]
                #print('input_image = ', input_image.shape)
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
            else:        
                pass

import argparse

if __name__ == "__main__":
    #device = 'MYRIAD'
    device = 'GPU'
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')    
    args.add_argument('-m', '--model', required=False, type=str, help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-d', '--device', required=False, default=device, type=str,help='Device name.')
    args.add_argument('-p', '--pre_api', required=False, default=True, type=bool, help='Device name.')
    args.add_argument('-bs', '--batchsize', required=False, default=1, type=int, help='Batch size.')
    args.add_argument('-n', '--nireq', required=False, default=2, type=int,help='number of infer request.')
    args.add_argument('-is', '--im_size', required=False, default=320, type=int, help='Input Image Size')
    args = parser.parse_args()
    #Convert Pytorch to ONNX
    #python export.py --weights 3dshape_tl_tiny_640.pt --img-size 320 --device cpu
    #python export.py --weights 3dshape_tl_tiny_640.pt --img-size 640 --device cpu
    
    args.model = f"/home/installer/yolov7/shapes_3d_{args.im_size}.onnx"
    #args.model = f"/home/installer/yolov7/old_models/yolov7_tiny_3dshapes.onnx"
    yolov7_detector= YOLOV7_OPENVINO(args.model, args.device, args.pre_api, args.batchsize, args.nireq, args.im_size)
    yolov7_detector.infer_cam()
    