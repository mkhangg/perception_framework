import pyrealsense2 as rs
import numpy as np

'''
Intrincs and Exrinsic information available from command line with RealSense SDK
$ rs-enumerate-devices -c
'''
config = rs.config()
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 640, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 640, rs.format.bgr8, 30)

decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

try:
    print("Getting color intrinsics")
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print("intr", intr)
    fx = float(intr.fx) # Focal length of x
    fy = float(intr.fy) # Focal length of y
    ppx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
    ppy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
    axs = 0.0 # Axis skew

    k_d435i = np.array([[fx, axs, ppx],
                        [0.0, fy, ppy],
                        [0.0, 0.0, 1.0]])
    print(intr)
    print()
    print(k_d435i)
    print()
    print("fx: ",intr.fx)
    print("fy: ",intr.fy)
    print("cx: ",intr.ppx)
    print("cy: ",intr.ppy)

    print()

    print("Getting depth intrinsics")
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx = float(intr.fx) # Focal length of x
    fy = float(intr.fy) # Focal length of y
    ppx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
    ppy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
    axs = 0.0 # Axis skew

    k_d435i = np.array([[fx, axs, ppx],
                        [0.0, fy, ppy],
                        [0.0, 0.0, 1.0]])
    print(intr)
    print()
    print(k_d435i)
    print()
    print("fx: ",intr.fx)
    print("fy: ",intr.fy)
    print("cx: ",intr.ppx)
    print("cy: ",intr.ppy)

finally:

    pipeline.stop()
    exit(0)