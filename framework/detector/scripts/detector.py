#!/usr/bin/env python3

import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.runtime import Core, get_version

sys.path.append(str(Path(__file__).resolve().parents[0] / 'common/python'))

print("Append:", str(Path(__file__).resolve().parents[0] / 'common/python'))

from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier

import monitors
from helpers import resolution
from images_capture import open_images_capture

from openvino.model_zoo.model_api.models import OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import pyrealsense2 as rs

#log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG |log.INFO | log.WARN, stream=sys.stdout)
log.basicConfig(level=log.DEBUG )

DEVICE_KINDS = ['CPU', 'GPU', 'MYRIAD', 'HETERO', 'HDDL']

fd_model="face-detection-0204"
lm_model="landmarks-regression-retail-0009"
id_model="face-reidentification-retail-0095"

class Args:
    input = 4
    loop = False
    output = None
    output_limit = 1000
    output_resolution = None
    no_show = False      #######
    crop_size = (0,0)
    match_algo = 'HUNGARIAN'
    u = ""
    utilization_monitors = ''

    fg = "/home/installer/faces"
    run_detector = False
    allow_grow = False

    m_fd = f"/home/installer/intel/{fd_model}/FP16/{fd_model}.xml" 
    m_lm = f"/home/installer/intel/{lm_model}/FP16/{lm_model}.xml"
    m_reid = f"/home/installer/intel/{id_model}/FP16/{id_model}.xml"
    fd_input_size = (0,0)

    d_fd = "GPU"
    d_lm = "GPU"
    d_reid = "GPU"
    verbose = True
    t_fd = 0.6
    t_id = 0.3
    exp_r_fd = 1.15




class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]
        #return rois


def draw_detections(frame, frame_processor, detections, output_transform):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        text = 'Hi ' + text
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %d%%' % (int(100.0 * (1 - identity.distance)))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame

def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]

import signal
import sys
b_run = True
def signal_handler(sig, frame):
    global b_run
    print('You pressed Ctrl+C!')
    b_run = False
    #sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    pub = rospy.Publisher('cameras/detector', Image, queue_size=10)
    pub_baxter = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=1)
    rospy.init_node('camera_detector', anonymous=True)
    br = CvBridge()
    #args = build_argparser().parse_args()
    args = Args()

    #cap = open_images_capture(args.input, args.loop)

    pipeline = rs.pipeline()
    config = rs.config()
    fps = 30
    ratio = 1.0
    width, height = int(640*ratio), int(480*ratio)
    #width, height = 1280, 720 #from cv2
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)


    frame_processor = FrameProcessor(args)

    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None
    if args.crop_size[0] > 0 and args.crop_size[1] > 0:
        input_crop = np.array(args.crop_size)
    elif not (args.crop_size[0] == 0 and args.crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')
    video_writer = cv2.VideoWriter()

    global b_run
    while b_run:
        start_time = perf_counter()
        #frame = cap.read()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break
        if input_crop:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
            #if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
            #                                         cap.fps(), output_resolution):
            #    raise RuntimeError("Can't open video writer")

        detections = frame_processor.process(frame)
        presenter.drawGraphs(frame)
        frame = draw_detections(frame, frame_processor, detections, output_transform)
        metrics.update(start_time, frame)

        frame_num += 1

        #if video_writer.isOpened() and (args.output_limit <= 0 or frame_num <= args.output_limit):
        #    video_writer.write(frame)

        if not args.no_show:
            if not rospy.is_shutdown():
                #print("Publish frame #", frame_num)
                #ros_frame = br.cv2_to_imgmsg(frame, encoding="bgr8")
                #pub.publish(ros_frame)
                resized = cv2.resize(frame, (1024, 600), interpolation = cv2.INTER_AREA)
                ros_frame2 = br.cv2_to_imgmsg(resized, encoding="bgr8")
                pub_baxter.publish(ros_frame2)
            
            # cv2.imshow('Face recognition', frame)
            # key = cv2.waitKey(1)
            # if key in {ord('q'), ord('Q'), 27}:
            #    break
            # presenter.handleKey(key)

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)

    print("Exiting main...")
    #rospy.signal_shutdown("")

if __name__ == '__main__':
    sys.exit(main() or 0)
