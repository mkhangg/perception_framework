#! usr/bin/python

# Import libraries
import cv2

# Camera parameters
fps = 14
width, height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))

for j in range(1, 171):
    img = cv2.imread('demo_' + str(j) + '.jpg')
    video.write(img)

cv2.destroyAllWindows()
video.release()
