## PerFc: An Efficient 2D and 3D Perception Software-Hardware Framework for Mobile Cobot

[Project Page](https://github.com/tuantdang/perception_framework) | [Paper]() | [Video Demo](https://www.youtube.com/watch?v=q4oz9Rixbzs&ab_channel=TuanDang)

## Author Information
- [Tuan Dang, UT Arlington, USA](https://www.tuandang.info/)
- [Khang Nguyen, UT Arlington, USA](https://mkhangg.com/)
- [Manfred Huber, UT Arlington, USA](https://www.uta.edu/academics/faculty/profile?username=huber)

 <p align="center">
<img src="images/fw.png" alt="" width="800"/>
</p>

## Abstract
In this work, we present an end-to-end software-hardware framework that supports both conventional hardware and software components and integrates machine learning object detectors without requiring an additional dedicated graphic processor unit (GPU). We first design our framework to achieve real-time performance on the robot system, guarantee configuration optimization, and concentrate on code reusability. We then mathematically model and utilize our transfer learning strategies for 2D object detection and fuse them into depth images for 3D depth estimation. Lastly, we systematically test the proposed framework and method on the Baxter robot with two 7-DOF arms and a four-wheel mobility base. The results show that the robot achieves real-time performance while executing other tasks (map building, localization, navigation, object detection, arm moving, and grasping) simultaneously with available hardware like **Intel onboard GPUs** on distributed computers. Also, to comprehensively control, program, and monitor the robot system, we design and introduce an end-user application.

 
## Graphic User Interface
<p align="center">
<img src="images/gui2.png" alt="" width="900"/>
</p>
 
 - Remote Control Co-bot using wireless connection (TCP/IP)
- Basic Control: Tuck/Untuck, Enable/Disable
- Joint Teaching
- World Position Monitor and Transformation
- Base control: Move Left/Right/Backward/Forward, Turn Left/Right
- Hands control
- Python download and execute on Robot


 ## Results
<p align="center">
<img src="images/results.png" alt="" width="900"/>
</p>


## Software Requirements
- Run distributed computers 
- Perform Machine Learning tasks using Onboard Intel GPU
- ROS Indigo Or Melodic
- Ubuntu 16.04 or 20.04 
- Python 2.7, 3.7 or above
- C/C++
- OpenVINO

## Acknowledgments


 
## Citing
```
To be included!
```


