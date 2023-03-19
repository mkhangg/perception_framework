import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
print("Environment Ready")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
#cfg.enable_device_from_file("../object_detection.bag")
cfg.enable_stream(rs.stream.color)
cfg.enable_stream(rs.stream.depth)
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipe.wait_for_frames()
  
# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()
print("Frames Captured")

color = np.asanyarray(color_frame.get_data())
colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_depth_frame = frameset.get_depth_frame()
colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())


# Standard OpenCV boilerplate for running the net:
height, width = color.shape[:2]
expected = 480
aspect = width / (height*1.0)
resized_image = cv2.resize(color, (round(expected * aspect), expected))
crop_start = round(expected * (aspect - 1) / 2)
crop_img = resized_image[0:expected, crop_start:crop_start+expected]
print("color = ", color.shape)
print("crop_img = ", crop_img.shape)
net = cv2.dnn.readNetFromCaffe("/home/installer/models/MobileNetSSD_deploy.prototxt", "/home/installer/models/MobileNetSSD_deploy.caffemodel")
inScaleFactor = 0.007843
meanVal       = 127.53
classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor")

blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
net.setInput(blob, "data")
detections = net.forward("detection_out")

label = detections[0,0,0,1]
conf  = detections[0,0,0,2]
xmin  = detections[0,0,0,3]
ymin  = detections[0,0,0,4]
xmax  = detections[0,0,0,5]
ymax  = detections[0,0,0,6]

#print("detections = ", detections[0,0,0,:])

className = classNames[int(label)]

cv2.rectangle(crop_img, (int(xmin * expected), int(ymin * expected)), (int(xmax * expected), int(ymax * expected)), (255, 0, 0), 2)
cv2.putText(crop_img, className, (int(xmin * expected), int(ymin * expected) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0))
plt.imshow(crop_img)
plt.show()

plt.figure(figsize=(12.2, 2.6))

scale = height / expected
#print("scale = ", scale)
xmin_depth = int((xmin * expected + crop_start) * scale)
ymin_depth = int((ymin * expected) * scale)
xmax_depth = int((xmax * expected + crop_start) * scale)
ymax_depth = int((ymax * expected) * scale)

cv2.rectangle(colorized_depth, (xmin_depth, ymin_depth), (xmax_depth, ymax_depth), (255, 0, 0), 2)
print("colorized_depth = ", colorized_depth.shape)
cv2.rectangle(color, (xmin_depth, ymin_depth), (xmax_depth, ymax_depth), (0, 255, 0), 2)
cv2.putText(color, className, (xmin_depth, int((ymin_depth+ymax_depth)/2.0)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

#Get raw depth data
depth = np.asanyarray(aligned_depth_frame.get_data())
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# Boundingbox depth data:
print(xmin_depth, xmax_depth, ymin_depth, ymax_depth)
depth1 = depth[ymin_depth:ymax_depth,xmin_depth:xmax_depth].astype(float)
print("depth_scale = ", depth_scale, ",depth = ", depth.shape, ", depth1 = ", depth1.shape)
depth1 = depth1 * depth_scale
dist,_,_,_ = cv2.mean(depth1)
print("Detected [1] a {0} {1:.3} meters away.".format(className, dist))

#Box from center
x_center = int((xmin_depth + xmax_depth)/2.0)
y_center = int((ymin_depth + ymax_depth)/2.0)
box_size = 20
depth2 = depth[y_center-box_size:y_center+box_size,x_center-box_size:x_center+box_size].astype(float)
print("depth_scale = ", depth_scale, ",depth = ", depth.shape, ", depth2 = ", depth2.shape)
depth2 = depth2 * depth_scale
dist,_,_,_ = cv2.mean(depth2)
print("Detected [2] a {0} {1:.3} meters away.".format(className, dist))
cv2.rectangle(colorized_depth, (x_center-box_size, y_center-box_size), (x_center+box_size, y_center+box_size), (0, 255, 255), 2)

#Show images
images = np.hstack((color, colorized_depth))
plt.imshow(images)
plt.show()